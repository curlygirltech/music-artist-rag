import { scrapeMultipleArtists } from './scraper.js';
import { writeFileSync } from 'node:fs';

async function test() {
    const artists = [
    'Beyoncé', 
    'Snoh Aalegra',
    'Bad Bunny',
    'Cardi B',
    'Solange Knowles',
    'SZA',
    'Tyler, the Creator',
    'Burna Boy',
    'Rosalía', 
    'Drake'
];
   console.log('🎵 Starting to scrape artist data...\n');
  const allData = await scrapeMultipleArtists(artists);
  
  // Save to a JSON file so we can use it later
  writeFileSync('artist-data.json', JSON.stringify(allData, null, 2));
  
  console.log('\n✅ Done! Scraped', allData.length, 'artists');
  console.log('📁 Data saved to artist-data.json');
}

test();