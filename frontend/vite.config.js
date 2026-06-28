import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'icons/*.png'],
      manifest: {
        name: 'OmniDiag — Clinical Decision Support',
        short_name: 'OmniDiag',
        description: 'Multi-disease AI diagnostic platform for clinical settings',
        theme_color: '#0f172a',
        background_color: '#0f172a',
        display: 'standalone',
        orientation: 'portrait-primary',
        scope: '/',
        start_url: '/',
        icons: [
          {
            src: '/icons/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable',
          },
          {
            src: '/icons/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable',
          },
        ],
      },
      workbox: {
        // Raise limit to 4 MiB to accommodate @react-pdf/renderer bundle
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
        // Pre-cache the app shell
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        // Cache last 5 API predict/explain responses (network-first)
        runtimeCaching: [
          {
            urlPattern: /\/api\/v4\/.+\/(predict|explain)$/,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'omnidiag-api-cache',
              expiration: { maxEntries: 5, maxAgeSeconds: 60 * 60 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            urlPattern: /\/api\/v4\/.+\/schema$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'omnidiag-schema-cache',
              expiration: { maxEntries: 10, maxAgeSeconds: 60 * 60 * 24 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
    }),
  ],
})
