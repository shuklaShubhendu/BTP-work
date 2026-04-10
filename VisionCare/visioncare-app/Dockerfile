# ── Multi-stage build ──────────────────────────────────────────────────────
# Stage 1: Build React app
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

# ── Stage 2: Serve with nginx ───────────────────────────────────────────────
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config: redirect all routes to index.html (React Router SPA)
RUN echo 'server { \
    listen 80; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    # Proxy API calls to backend \
    location /api/ { \
        proxy_pass http://backend:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
    location /images/ { \
        proxy_pass http://backend:8000; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
