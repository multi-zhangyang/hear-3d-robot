import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765"
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          minSize: 20 * 1024,
          // No maxSize: these packages have circular imports between their own
          // modules, and slicing a group into size-capped fragments emits them
          // in an order where a module is called before its definition has been
          // evaluated. The production application then fails on load with
          // "s is not a function" while the dev server works normally.
          groups: [
            {
              name: "three",
              test: /node_modules[\\/]three[\\/]/,
              priority: 30,
              entriesAware: true
            },
            {
              name: "react",
              test: /node_modules[\\/](?:react|react-dom|scheduler)[\\/]/,
              priority: 30
            },
            {
              name: "antd",
              test: /node_modules[\\/](?:antd|@ant-design|@rc-component|rc-[^\\/]+)[\\/]/,
              priority: 20,
              entriesAware: true
            },
            {
              name: "vendor",
              test: /node_modules[\\/]/,
              priority: 1,
              entriesAware: true
            }
          ]
        }
      }
    }
  }
});
