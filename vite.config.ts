import { createReadStream } from "node:fs";
import { access, copyFile, mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Tauri expects a fixed dev port and disables the browser auto-open.
const host = process.env.TAURI_DEV_HOST;

const MODEL_FILES = [
  "yolo26s-manga-seg.onnx",
  "ppocrv6-det.onnx",
  "ppocrv6-rec.onnx",
  "ppocrv6_dict.txt",
] as const;

function modelAssetsPlugin(): Plugin {
  const sourceDir = resolve("models");
  let outputDir = resolve("dist");
  let runningTests = false;

  const availableFiles = async () => {
    const files: string[] = [];
    for (const file of MODEL_FILES) {
      try {
        await access(resolve(sourceDir, file));
        files.push(file);
      } catch {
        // Missing weights are allowed for local demo builds.
      }
    }
    return files;
  };

  return {
    name: "manga-model-assets",
    configResolved(config) {
      outputDir = resolve(config.root, config.build.outDir);
      runningTests = config.mode === "test";
    },
    configureServer(server) {
      server.middlewares.use("/models", async (req, res, next) => {
        const requestPath = decodeURIComponent((req.url ?? "/").split("?", 1)[0])
          .replace(/^\/+/, "");

        if (requestPath === "manifest.json") {
          const files = await availableFiles();
          if (!MODEL_FILES.every((file) => files.includes(file))) {
            res.statusCode = 404;
            res.end();
            return;
          }
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ version: 1, files }));
          return;
        }

        if (!MODEL_FILES.includes(requestPath as (typeof MODEL_FILES)[number])) {
          next();
          return;
        }

        const source = resolve(sourceDir, requestPath);
        try {
          await access(source);
        } catch {
          next();
          return;
        }

        res.setHeader(
          "Content-Type",
          requestPath.endsWith(".txt") ? "text/plain; charset=utf-8" : "application/octet-stream"
        );
        if (req.method === "HEAD") {
          res.statusCode = 200;
          res.end();
          return;
        }
        createReadStream(source).pipe(res);
      });
    },
    async closeBundle() {
      if (runningTests) return;
      const files = await availableFiles();
      const requireModels = process.env.MANGA_BUNDLE_MODELS === "1";
      const missing = MODEL_FILES.filter((file) => !files.includes(file));
      if (requireModels && missing.length > 0) {
        throw new Error(`Required release model assets are missing: ${missing.join(", ")}`);
      }
      if (!MODEL_FILES.every((file) => files.includes(file))) return;

      const modelOutputDir = resolve(outputDir, "models");
      await mkdir(modelOutputDir, { recursive: true });
      await Promise.all(
        files.map((file) => copyFile(resolve(sourceDir, file), resolve(modelOutputDir, file)))
      );
      await writeFile(
        resolve(modelOutputDir, "manifest.json"),
        JSON.stringify({ version: 1, files }, null, 2),
        "utf8"
      );
    },
  };
}

export default defineConfig({
  plugins: [react(), modelAssetsPlugin()],
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
