import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri expects a fixed dev port and disables the browser auto-open.
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  // Prevent vite from obscuring rust errors
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? { protocol: "ws", host, port: 1421 }
      : undefined,
    watch: {
      // Don't watch the rust backend
      ignored: ["**/src-tauri/**"],
    },
  },
  // onnxruntime-web ships wasm assets; make sure they are bundled/served.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  test: {
    environment: "node",
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
} as any);
