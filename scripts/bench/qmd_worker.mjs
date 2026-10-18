// Pinned QMD 2.8.3 SDK. stdout is a private JSONL protocol, never document text.
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline';
console.log = console.error;
const [modulePath, dbPath, corpus, models] = process.argv.slice(2);
const { createStore } = await import(pathToFileURL(modulePath));
const { setDefaultLlamaCpp } = await import(new URL('./llm.js', pathToFileURL(modulePath)));
const store = await createStore({ dbPath, config: {
  collections: { bench: { path: corpus, pattern: '*.txt' } },
  models: {
    embed: `${models}/embeddinggemma-300M-Q8_0.gguf`,
    rerank: `${models}/qwen3-reranker-0.6b-q8_0.gguf`,
    generate: `${models}/qmd-query-expansion-1.7B-q4_k_m.gguf`,
  },
} });
// 2.8.3's token chunker uses the module singleton, even for SDK stores. Point it
// at this store's explicitly local models, as the CLI does for its model config.
setDefaultLlamaCpp(store.internal.llm);
try {
  for await (const line of createInterface({ input: process.stdin })) {
    const request = JSON.parse(line);
    const start = performance.now();
    let response;
    if (request.op === 'index') {
      await store.update();
      console.error('QMD collection updated; initializing local runtime');
      if (request.vectors && (await store.internal.llm.getDeviceInfo({ allowBuild: false })).gpu !== 'metal') {
        throw new Error('QMD did not use the requested Metal runtime');
      }
      console.error('QMD local runtime ready; embedding');
      if (request.vectors) await store.embed({ chunkStrategy: 'regex' });
      response = { status: await store.getStatus() };
    } else {
      if (request.mode !== 'bm25' && (await store.internal.llm.getDeviceInfo({ allowBuild: false })).gpu !== 'metal') {
        throw new Error('QMD did not use the requested Metal runtime');
      }
      const options = { limit: 200, collection: 'bench' };
      let hits;
      if (request.mode === 'bm25') hits = await store.searchLex(request.q, options);
      else if (request.mode === 'vector') hits = await store.searchVector(request.q, options);
      else hits = await store.search({ ...options, query: request.q,
        candidateLimit: 200, rerank: request.mode === 'full', chunkStrategy: 'regex' });
      response = { hits: hits.map(hit => ({
        path: hit.filepath ?? hit.file, score: hit.score,
        offset: hit.chunkPos ?? hit.bestChunkPos ?? null,
      })) };
    }
    process.stdout.write(JSON.stringify({ ...response, id: request.id,
      elapsed_ms: performance.now() - start }) + '\n');
  }
} finally { await store.close(); }
