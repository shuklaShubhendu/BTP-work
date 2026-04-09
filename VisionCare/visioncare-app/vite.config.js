import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Allow all external hosts (ngrok, localtunnel, etc.)
    allowedHosts: 'all',
  },
})
