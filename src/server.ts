import * as http from 'node:http';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { chatWithContext } from './chat.js';

const PORT = 3000;

const server = http.createServer(async (req, res) => {
  // Serve the HTML page
  if (req.method === 'GET' && req.url === '/') {
    const htmlPath = path.join(process.cwd(), 'public', 'index.html');
    try {
      const html = fs.readFileSync(htmlPath, 'utf-8');
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Error loading page');
    }
    return;
  }

  // Handle chat API
  if (req.method === 'POST' && req.url === '/chat') {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', async () => {
      try {
        const { message } = JSON.parse(body);
        const { response, isNewArtist } = await chatWithContext(message);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ response, isNewArtist }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Failed to process request' }));
      }
    });
    return;
  }

  // 404 for other routes
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
