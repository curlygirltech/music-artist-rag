import Anthropic from '@anthropic-ai/sdk';
import { QdrantClient } from '@qdrant/qdrant-js';
import { VoyageAIClient } from 'voyageai';
import * as dotenv from 'dotenv';
import * as readline from 'node:readline';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { scrapeArtistData, type ArtistData } from './scraper.js';
import { addSingleArtist } from './embeddings.js';

dotenv.config();

const ARTIST_DATA_PATH = path.join(process.cwd(), 'artist-data.json');

// Persist a new artist to the local JSON file
function persistArtistToJson(artistData: ArtistData): void {
  try {
    // Read existing data
    let existingArtists: ArtistData[] = [];
    if (fs.existsSync(ARTIST_DATA_PATH)) {
      const rawData = fs.readFileSync(ARTIST_DATA_PATH, 'utf-8');
      existingArtists = JSON.parse(rawData);
    }

    // Check if artist already exists (by name, case-insensitive)
    const alreadyExists = existingArtists.some(
      (a) => a.name.toLowerCase() === artistData.name.toLowerCase()
    );

    if (!alreadyExists) {
      existingArtists.push(artistData);
      fs.writeFileSync(ARTIST_DATA_PATH, JSON.stringify(existingArtists, null, 2));
      console.log(`📝 Saved "${artistData.name}" to artist-data.json`);
    }
  } catch (error) {
    console.error('⚠️ Could not persist artist to JSON:', error);
    // Non-fatal error - the artist is still in Qdrant
  }
}

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

// Clean up an extracted artist name by removing possessives and trailing topic words
function cleanArtistName(rawName: string): string {
  // Remove possessive endings and everything after
  // e.g., "Taylor Swift's career" -> "Taylor Swift"
  let cleaned = rawName.replace(/'s\s+.+$/i, '');
  cleaned = cleaned.replace(/'s$/i, '');

  // Remove common trailing topic words
  const trailingWords = [
    'career', 'songs', 'albums', 'discography', 'awards', 'biography',
    'bio', 'history', 'life', 'music', 'genre', 'genres', 'hits',
    'singles', 'records', 'achievements', 'background', 'story'
  ];

  const trailingPattern = new RegExp(`\\s+(${trailingWords.join('|')})\\s*$`, 'i');
  cleaned = cleaned.replace(trailingPattern, '');

  // Clean up punctuation and extra whitespace
  cleaned = cleaned.replace(/[?.,!]$/, '').trim();

  return cleaned;
}

// Convert a name to title case (e.g., "taylor swift" -> "Taylor Swift")
function toTitleCase(str: string): string {
  return str
    .toLowerCase()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Extract artist name from a user query
function extractArtistName(query: string): string | null {
  console.log(`\n🔬 [DEBUG] Extracting artist from: "${query}"`);

  // Check for quoted names first (e.g., "Tell me about 'Taylor Swift'")
  const quotedMatch = query.match(/["']([^"']+)["']/);
  if (quotedMatch && quotedMatch[1]) {
    const result = toTitleCase(cleanArtistName(quotedMatch[1]));
    console.log(`   ✓ Found quoted name: "${result}"`);
    return result;
  }
  console.log(`   ✗ No quoted name found`);

  // Common patterns to extract artist names
  const patterns = [
    { name: 'tell me about', regex: /(?:tell me about|who is|what about|information on|info on)\s+(.+?)(?:\?|$)/i },
    { name: 'what X does', regex: /(?:what (?:genres?|music|songs?|albums?|awards?) does)\s+(.+?)\s+(?:perform|make|have|play|sing)/i },
    { name: 'what X did', regex: /(?:what (?:genres?|music|songs?|albums?|awards?) (?:does|did))\s+(.+?)\s+/i },
    { name: 'does/did/is', regex: /(?:does|did|is|was|has|have)\s+(.+?)\s+(?:won|win|have|perform|make|play|sing)/i },
  ];

  for (const { name: patternName, regex } of patterns) {
    const match = query.match(regex);
    if (match && match[1]) {
      const rawMatch = match[1];
      const cleaned = cleanArtistName(rawMatch);
      console.log(`   Pattern "${patternName}" matched: "${rawMatch}" → cleaned: "${cleaned}"`);
      // Accept the match if it has at least 2 characters (removed capital letter requirement)
      if (cleaned.length >= 2) {
        const result = toTitleCase(cleaned);
        console.log(`   ✓ Returning: "${result}"`);
        return result;
      }
    }
  }
  console.log(`   ✗ No pattern matches`);

  // Fallback: look for sequences of 2+ words that could be a name
  // Try capitalized words first
  const capitalizedWords = query.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g);
  console.log(`   Capitalized words found: ${JSON.stringify(capitalizedWords)}`);

  if (capitalizedWords && capitalizedWords.length > 0) {
    const commonWords = ['Tell', 'What', 'Who', 'How', 'When', 'Where', 'Why', 'Can', 'Does', 'Did', 'Is', 'Are', 'The'];
    const filtered = capitalizedWords.filter(w => !commonWords.includes(w));
    console.log(`   After filtering common words: ${JSON.stringify(filtered)}`);
    if (filtered.length > 0) {
      const result = cleanArtistName(filtered[0] ?? '');
      console.log(`   ✓ Returning fallback: "${result}"`);
      return result;
    }
  }

  // Last resort: if query is short (likely just a name), use the whole thing
  const trimmedQuery = query.trim().replace(/[?.,!]$/, '');
  if (trimmedQuery.length >= 2 && trimmedQuery.split(' ').length <= 4) {
    const result = toTitleCase(cleanArtistName(trimmedQuery));
    console.log(`   ✓ Using entire query as artist name: "${result}"`);
    return result;
  }

  console.log(`   ✗ No artist detected`);
  return null;
}

// Check if an artist exists in the local JSON file
function isArtistInJson(artistName: string): boolean {
  try {
    if (!fs.existsSync(ARTIST_DATA_PATH)) {
      return false;
    }
    const rawData = fs.readFileSync(ARTIST_DATA_PATH, 'utf-8');
    const artists: ArtistData[] = JSON.parse(rawData);
    return artists.some(
      (a) => a.name.toLowerCase() === artistName.toLowerCase()
    );
  } catch {
    return false;
  }
}

// Search Qdrant for relevant context
async function searchContext(query: string, limit: number = 5): Promise<{ context: string; isNewArtist: boolean }> {
  let isNewArtist = false;

  try {
    console.log('\n🔍 Processing query...');

    // Step 1: Extract artist name from query
    const requestedArtist = extractArtistName(query);
    console.log(`🎤 Detected artist: ${requestedArtist || '(none)'}`);

    // Step 2: If we detected an artist, check if they're in our JSON
    // If not, scrape and add them BEFORE searching Qdrant
    if (requestedArtist) {
      const artistExists = isArtistInJson(requestedArtist);
      console.log(`📋 "${requestedArtist}" in artist-data.json: ${artistExists ? 'YES ✓' : 'NO ✗'}`);

      if (!artistExists) {
        console.log(`🆕 Scraping Wikipedia for "${requestedArtist}"...`);

        try {
          // Scrape Wikipedia for the artist
          const artistData = await scrapeArtistData(requestedArtist);
          console.log(`📄 Scraped: "${artistData.name}" (${artistData.genres.join(', ') || 'no genres'})`);

          // Add to the vector database
          await addSingleArtist(artistData);
          console.log(`💾 Added to Qdrant`);

          // Persist to local JSON
          persistArtistToJson(artistData);

          isNewArtist = true;
        } catch (scrapeError) {
          console.error(`❌ Failed to scrape "${requestedArtist}":`, scrapeError);
          // Continue anyway - we'll search with what we have
        }
      }
    }

    // Step 3: Now do a SINGLE Qdrant search
    console.log(`🔍 Searching Qdrant...`);
    const queryEmbedding = await getQueryEmbedding(query);

    const searchResults = await qdrant.search(COLLECTION_NAME, {
      vector: queryEmbedding,
      limit: limit,
      with_payload: true,
    });

    // Step 4: If we're looking for a specific artist, filter results to that artist
    let filteredResults = searchResults;
    if (requestedArtist) {
      const artistResults = searchResults.filter((result) => {
        const payload = result.payload as { artistName: string };
        return payload.artistName.toLowerCase().includes(requestedArtist.toLowerCase()) ||
               requestedArtist.toLowerCase().includes(payload.artistName.toLowerCase());
      });

      // Only use filtered results if we found some for the requested artist
      if (artistResults.length > 0) {
        filteredResults = artistResults;
        console.log(`✅ Found ${filteredResults.length} chunks for "${requestedArtist}"`);
      } else {
        console.log(`⚠️ No chunks found for "${requestedArtist}", using general results`);
      }
    } else {
      console.log(`✅ Found ${filteredResults.length} relevant chunks`);
    }

    // Step 5: Return results
    if (filteredResults.length === 0) {
      return {
        context: 'No relevant information found for this query.',
        isNewArtist: false,
      };
    }

    // Build context string
    const context = filteredResults
      .map((result, index) => {
        const payload = result.payload as { artistName: string; text: string };
        return `[Source ${index + 1} - ${payload.artistName}]\n${payload.text}\n---`;
      })
      .join('\n\n');

    return { context, isNewArtist };

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
You will be provided with relevant context about artists.
Use this context to answer the user's question accurately and concisely.
If the context doesn't contain the answer, say so politely.`;

    const userPrompt = `Context:
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
        const { response } = await chatWithContext(input);
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
  const { response } = await chatWithContext(question);
  console.log(`\n💬 Answer: ${response}\n`);
}

// Run chat loop if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  startChatLoop();
}