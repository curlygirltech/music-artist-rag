import Anthropic from '@anthropic-ai/sdk';
import { QdrantClient } from '@qdrant/qdrant-js';
import { readFileSync } from 'node:fs';
import * as dotenv from 'dotenv';
import { VoyageAIClient } from 'voyageai';
import type { ArtistData } from './scraper.js';

dotenv.config();

// Initialize clients
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY!,
});

const voyage = new VoyageAIClient({
  apiKey: process.env.VOYAGE_API_KEY!,
});

const COLLECTION_NAME = 'music-artists';

// Split text into chunks for embedding
export function chunkText(text: string, maxLength: number = 1000): string[] {
  const chunks: string[] = [];
  const paragraphs = text.split('\n\n');
  
  let currentChunk = '';
  
  for (const para of paragraphs) {
    if ((currentChunk + para).length > maxLength && currentChunk.length > 0) {
      chunks.push(currentChunk.trim());
      currentChunk = para;
    } else {
      currentChunk += '\n\n' + para;
    }
  }
  
  if (currentChunk.trim().length > 0) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks;
}

// Get embeddings from Voyage AI
export async function getEmbedding(text: string): Promise<number[]> {
  try {
    const result = await voyage.embed({
      input: text,
      model: 'voyage-3-lite',
    });
    if (!result.data || !result.data[0] || !result.data[0].embedding) {
      throw new Error('No embedding data returned');
    }
    return result.data[0].embedding as number[];
  } catch (error) {
    console.error('Error getting embedding:', error);
    throw error;
  }
}

// Batch embeddings
export async function getBatchEmbeddings(texts: string[]): Promise<number[][]> {
  try {
    console.log(`  🔄 Getting embeddings for ${texts.length} chunks...`);
    
    const result = await voyage.embed({
      input: texts,
      model: 'voyage-3-lite',
    });
      if (!result.data) {
      throw new Error('No embedding data returned');
    }
    return result.data.map(item => item.embedding as number[]);
  } catch (error) {
    console.error('Error getting batch embeddings:', error);
    throw error;
  }
}

// Create collection in Qdrant
async function createCollection(collectionName: string) {
  try {
    // Check if collection exists
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(c => c.name === collectionName);
    
    if (exists) {
      console.log(`📦 Collection "${collectionName}" already exists, deleting...`);
      await qdrant.deleteCollection(collectionName);
    }
    
    console.log(`📦 Creating collection "${collectionName}"...`);
    await qdrant.createCollection(collectionName, {
      vectors: {
        size: 512, // voyage-3-lite produces 512-dimensional embeddings
        distance: 'Cosine', // Similarity metric
      },
    });
    
    console.log('✅ Collection created!');
  } catch (error) {
    console.error('Error creating collection:', error);
    throw error;
  }
}

// Store artist data in Qdrant
export async function storeArtistData(collectionName: string = 'music-artists') {
  try {
    // Step 1: Read the scraped data
    console.log('📖 Reading artist data...');
    const artistDataRaw = readFileSync('artist-data.json', 'utf-8');
    const artistData: ArtistData[] = JSON.parse(artistDataRaw);
    
    // Step 2: Create collection
    await createCollection(collectionName);
    
    // Step 3: Process each artist
    console.log('\n🔄 Processing artists and creating embeddings...\n');
    
    let pointId = 0;
    const points: any[] = [];
    
    for (const artist of artistData) {
      console.log(`Processing ${artist.name}...`);
      
      // Chunk the content
      const chunks = chunkText(artist.content);
      console.log(`  - Created ${chunks.length} chunks`);
      
      // Get embeddings in batch (more efficient!)
      const embeddings = await getBatchEmbeddings(chunks);
      console.log(`  ✅ Got embeddings for ${chunks.length} chunks`);
      
      // Create points for Qdrant
      for (let i = 0; i < chunks.length; i++) {
        points.push({
          id: pointId++,
          vector: embeddings[i],
          payload: {
            artistName: artist.name,
            text: chunks[i],
            birthDate: artist.birthDate,
            genres: artist.genres,
          },
        });
      }
    }
    
    // Step 4: Upload to Qdrant in batches
    console.log(`\n📤 Uploading ${points.length} vectors to Qdrant...`);
    
    const batchSize = 100;
    for (let i = 0; i < points.length; i += batchSize) {
      const batch = points.slice(i, i + batchSize);
      await qdrant.upsert(collectionName, {
        wait: true,
        points: batch,
      });
      console.log(`  - Uploaded batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(points.length / batchSize)}`);
    }
    
    console.log('\n✅ All data stored in Qdrant!');
    console.log(`📊 Total vectors: ${points.length}`);
    console.log('\n🎉 Phase 2 Complete! Ready for retrieval and chat!');

  } catch (error) {
    console.error('❌ Error storing data:', error);
    throw error;
  }
}

// Add a single artist to the existing collection (for on-demand scraping)
export async function addSingleArtist(artistData: ArtistData): Promise<number> {
  try {
    console.log(`📥 Adding ${artistData.name}...`);

    // Step 1: Get current max point ID from collection
    const collectionInfo = await qdrant.getCollection(COLLECTION_NAME);
    const startId = collectionInfo.points_count ?? 0;

    // Step 2: Chunk the artist content
    const chunks = chunkText(artistData.content);
    console.log(`  - Created ${chunks.length} chunks`);

    // Step 3: Get embeddings for all chunks
    const embeddings = await getBatchEmbeddings(chunks);
    console.log(`  ✅ Got embeddings for ${chunks.length} chunks`);

    // Step 4: Create points with incrementing IDs
    const points = chunks.map((chunk, i) => {
      const vector = embeddings[i];
      if (!vector) {
        throw new Error(`Missing embedding for chunk ${i}`);
      }
      return {
        id: startId + i,
        vector,
        payload: {
          artistName: artistData.name,
          text: chunk,
          birthDate: artistData.birthDate,
          genres: artistData.genres,
        },
      };
    });

    // Step 5: Upsert to existing collection
    await qdrant.upsert(COLLECTION_NAME, {
      wait: true,
      points,
    });

    console.log(`✅ Added ${chunks.length} vectors for ${artistData.name}`);
    return chunks.length;

  } catch (error) {
    console.error(`❌ Error adding artist ${artistData.name}:`, error);
    throw error;
  }
}

// Test function
async function test() {
  await storeArtistData();
}

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  test();
}