import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [{ src: 'node_modules/@mediapipe/tasks-vision/wasm/*', dest: 'wasm' }],
    }),
  ],
  server: { host: '127.0.0.1', port: 2502 },
  preview: { host: '127.0.0.1', port: 2502 },
  build: { target: 'ES2022', sourcemap: true },
});
