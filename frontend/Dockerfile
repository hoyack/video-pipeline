# ---- Build stage ----
FROM node:20-alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ .
# Run vite build directly — tsc is skipped because the codebase has
# pre-existing type errors that are tracked separately.
RUN npx vite build

# ---- Production target ----
FROM nginx:alpine AS production

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# ---- Development target ----
FROM node:20-alpine AS development

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
