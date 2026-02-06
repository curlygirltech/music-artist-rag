import Anthropic from '@anthropic-ai/sdk';
import { QdrantClient } from '@qdrant/qdrant-js';
import { VoyageAIClient } from 'voyageai';
import * as dotenv from 'dotenv';
import * as readline from 'node:readline';
import { scrapeArtistData } from './scraper.js';
import { addSingleArtist } from './embeddings.js';

dotenv.config();

// Initialize clients
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY!,
});

const voyage = new VoyageAIClient({
  apiKey: process.env.VOYAGE_API_KEY!,
});

const COLLECTION_NAME = 'music-artists';

// Get embedding for a query
async function getQueryEmbedding(query: string): Promise<number[]> {
  const result = await voyage.embed({
    input: query,
    model: 'voyage-3-lite',
  });

  if (!result.data || !result.data[0]) {
    throw new Error('No embedding data returned');
  }

  return result.data[0].embedding as number[];
}

// Extract artist name from a user query
function extractArtistName(query: string): string | null {
  // Check for quoted names first (e.g., "Tell me about 'Taylor Swift'")
  const quotedMatch = query.match(/["']([^"']+)["']/);
  if (quotedMatch && quotedMatch[1]) {
    return quotedMatch[1];
  }

  // Common patterns to extract artist names
  const patterns = [
    /(?:tell me about|who is|what about|information on|info on)\s+(.+?)(?:\?|$)/i,
    /(?:what (?:genres?|music|songs?|albums?|awards?) does)\s+(.+?)\s+(?:perform|make|have|play|sing)/i,
    /(?:what (?:genres?|music|songs?|albums?|awards?) (?:does|did))\s+(.+?)\s+/i,
    /(?:does|did|is|was|has|have)\s+(.+?)\s+(?:won|win|have|perform|make|play|sing)/i,
  ];

  for (const pattern of patterns) {
    const match = query.match(pattern);
    if (match && match[1]) {
      // Clean up the match
      const name = match[1].trim().replace(/[?.,!]$/, '');
      // Validate it looks like a name (has capital letters)
      if (/[A-Z]/.test(name)) {
        return name;
      }
    }
  }

  // Fallback: look for sequences of capitalized words (likely proper nouns)
  const capitalizedWords = query.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g);
  if (capitalizedWords && capitalizedWords.length > 0) {
    // Filter out common words that might be capitalized at sentence start
    const commonWords = ['Tell', 'What', 'Who', 'How', 'When', 'Where', 'Why', 'Can', 'Does', 'Did', 'Is', 'Are', 'The'];
    const filtered = capitalizedWords.filter(w => !commonWords.includes(w));
    if (filtered.length > 0) {
      return filtered[0] ?? null;
    }
  }

  return null;
}

// Search Qdrant for relevant context
async function searchContext(query: string, limit: number = 3): Promise<{ context: string; isNewArtist: boolean }> {
  try {
    console.log('\n🔍 Searching for relevant information...');

    // Step 1: Convert query to embedding
    const queryEmbedding = await getQueryEmbedding(query);

    // Step 2: Search Qdrant for similar vectors
    const searchResults = await qdrant.search(COLLECTION_NAME, {
      vector: queryEmbedding,
      limit: limit,
      with_payload: true,
      score_threshold: 0.3, // Filter out very low confidence matches
    });

    // Step 3: Check if we need to scrape a new artist
    const topScore = searchResults[0]?.score ?? 0;
    const needsScraping = searchResults.length === 0 || topScore < 0.3;

    if (needsScraping) {
      const artistName = extractArtistName(query);
      if (artistName) {
        console.log(`🆕 Artist not found in knowledge base. Scraping ${artistName}...`);

        try {
          // Scrape Wikipedia for the artist
          const artistData = await scrapeArtistData(artistName);

          // Add to the vector database
          await addSingleArtist(artistData);

          // Retry the search with the new data
          const retryResults = await qdrant.search(COLLECTION_NAME, {
            vector: queryEmbedding,
            limit: limit,
            with_payload: true,
          });

          if (retryResults.length > 0) {
            console.log(`✅ Found ${retryResults.length} chunks after adding ${artistName}\n`);

            const context = retryResults
              .map((result, index) => {
                const payload = result.payload as { artistName: string; text: string };
                return `[Source ${index + 1} - ${payload.artistName}]\n${payload.text}\n---`;
              })
              .join('\n\n');

            return { context, isNewArtist: true };
          }
        } catch (scrapeError) {
          console.error(`❌ Failed to scrape ${artistName}:`, scrapeError);
          // Fall through to return no results message
        }
      }

      return {
        context: 'No relevant information found in the database.',
        isNewArtist: false,
      };
    }

    console.log(`✅ Found ${searchResults.length} relevant chunks\n`);

    // Build context string
    const context = searchResults
      .map((result, index) => {
        const payload = result.payload as { artistName: string; text: string };
        return `[Source ${index + 1} - ${payload.artistName}]\n${payload.text}\n---`;
      })
      .join('\n\n');

    return { context, isNewArtist: false };

  } catch (error) {
    console.error('Error searching context:', error);
    throw error;
  }
}

// Chat with Claude using retrieved context
export async function chatWithContext(userQuery: string): Promise<{ response: string; isNewArtist: boolean }> {
  try {
    // Step 1: Get relevant context from our vector database
    const { context, isNewArtist } = await searchContext(userQuery);

    // Step 2: Create a prompt for Claude with the context
    const systemPrompt = `You are a helpful assistant that answers questions about music artists.
You will be provided with relevant context from a database of artist information.
Use this context to answer the user's question accurately and concisely.
If the context doesn't contain the answer, say so politely.`;

    const userPrompt = `Context from database:
${context}

User question: ${userQuery}

Please answer the question based on the context provided above.`;

    // Step 3: Send to Claude
    console.log('🤖 Claude is thinking...\n');

    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1024,
      system: systemPrompt,
      messages: [
        {
          role: 'user',
          content: userPrompt,
        },
      ],
    });

    // Step 4: Extract and return the response
    const content = message.content[0];
    if (content && 'text' in content) {
      return { response: content.text, isNewArtist };
    }

    return { response: 'Sorry, I could not generate a response.', isNewArtist };

  } catch (error) {
    console.error('Error chatting with Claude:', error);
    throw error;
  }
}

// Interactive CLI chat loop
async function startChatLoop() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  
  console.log('\n🎵 Music Artist RAG Chat 🎵');
  console.log('================================\n');
  console.log('Ask me anything about your artists!');
  console.log('Type "exit" or "quit" to end the conversation.\n');
  
  const askQuestion = () => {
    rl.question('You: ', async (userInput) => {
      const input = userInput.trim();
      
      // Check for exit commands
      if (input.toLowerCase() === 'exit' || input.toLowerCase() === 'quit') {
        console.log('\n👋 Thanks for chatting! Goodbye!\n');
        rl.close();
        return;
      }
      
      // Skip empty inputs
      if (!input) {
        askQuestion();
        return;
      }
      
      try {
        // Get response from Claude with context
        const { response, isNewArtist } = await chatWithContext(input);
        if (isNewArtist) {
          console.log(`\n✨ Added new artist to knowledge base!\n`);
        }
        console.log(`\nAssistant: ${response}\n`);
      } catch (error) {
        console.error('\n❌ Error:', error);
      }
      
      // Ask next question
      askQuestion();
    });
  };
  
  askQuestion();
}

// Single question mode (for testing)
export async function askQuestion(question: string) {
  console.log(`\n📝 Question: ${question}`);
  const { response, isNewArtist } = await chatWithContext(question);
  if (isNewArtist) {
    console.log(`\n✨ Added new artist to knowledge base!`);
  }
  console.log(`\n💬 Answer: ${response}\n`);
}

// Run chat loop if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  startChatLoop();
}