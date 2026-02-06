import axios from 'axios';
import * as cheerio from 'cheerio';

// This interface defines what data we want about each artist
export interface ArtistData {
  name: string;
  bio: string;
  birthDate?: string;
  genres: string[];
  discography: string[];
  awards: string[];
  content: string; // Full text we'll use for RAG
}

// Function to scrape Wikipedia for an artist
export async function scrapeArtistData(artistName: string): Promise<ArtistData> {
  try {
    console.log(`🔍 Scraping data for ${artistName}...`);
    
    // Step 1: Get the Wikipedia page
    const searchUrl = `https://en.wikipedia.org/wiki/${artistName.replace(/ /g, '_')}`;
    const response = await axios.get(searchUrl, {
  headers: {
    'User-Agent': 'MusicArtistRAG/1.0 (Educational Project; jovonnecameron@example.com)'
  }
});

    const $ = cheerio.load(response.data);
   
    
    // Step 2: Extract the bio (first few paragraphs)
    const bioParas: string[] = [];
    $('#mw-content-text .mw-parser-output > p').each((i, elem) => {
      const text = $(elem).text().trim();
      if (text && i < 3) { // Get first 3 paragraphs
        bioParas.push(text);
      }
    });
    const bio = bioParas.join('\n\n');
    
    // Step 3: Extract birth date from infobox
    let birthDate = '';
    $('.infobox tr').each((i, elem) => {
      const header = $(elem).find('th').text().toLowerCase();
      if (header.includes('born')) {
        birthDate = $(elem).find('td').text().trim();
      }
    });
    
    // Step 4: Extract genres
    const genres: string[] = [];
    $('.infobox tr').each((i, elem) => {
      const header = $(elem).find('th').text().toLowerCase();
      if (header.includes('genre')) {
        $(elem).find('td a').each((j, link) => {
          genres.push($(link).text().trim());
        });
      }
    });
    
    // Step 5: Extract discography (albums)
    const discography: string[] = [];
    $('h2, h3').each((i, elem) => {
      const heading = $(elem).text().toLowerCase();
      if (heading.includes('discography') || heading.includes('albums')) {
        // Get the next list after the discography heading
        $(elem).nextAll('ul').first().find('li').each((j, item) => {
          const albumText = $(item).text().trim();
          if (albumText && j < 10) { // Limit to 10 albums
            discography.push(albumText);
          }
        });
      }
    });
    
    // Step 6: Extract awards
    const awards: string[] = [];
    $('h2, h3').each((i, elem) => {
      const heading = $(elem).text().toLowerCase();
      if (heading.includes('awards') || heading.includes('accolades')) {
        $(elem).nextAll('ul').first().find('li').each((j, item) => {
          const awardText = $(item).text().trim();
          if (awardText && j < 10) { // Limit to 10 awards
            awards.push(awardText);
          }
        });
      }
    });
    
    // Step 7: Create the full content for RAG
    // This is what we'll chunk and embed
    const content = `
Artist: ${artistName}

Biography:
${bio}

Birth Date: ${birthDate}

Genres: ${genres.join(', ')}

Discography:
${discography.join('\n')}

Awards and Recognition:
${awards.join('\n')}
    `.trim();
    
    console.log(`✅ Successfully scraped ${artistName}`);
    
    return {
      name: artistName,
      bio,
      birthDate,
      genres,
      discography,
      awards,
      content
    };
    
  } catch (error) {
    console.error(`❌ Error scraping ${artistName}:`, error);
    throw error;
  }
}

// Helper function to scrape multiple artists
export async function scrapeMultipleArtists(artistNames: string[]): Promise<ArtistData[]> {
  const results: ArtistData[] = [];
  
  for (const name of artistNames) {
    try {
      const data = await scrapeArtistData(name);
      results.push(data);
      // Be nice to Wikipedia - wait 1 second between requests
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (error) {
      console.error(`Failed to scrape ${name}, skipping...`);
    }
  }
  
  return results;
}