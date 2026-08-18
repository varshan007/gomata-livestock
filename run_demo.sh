#!/bin/bash

# GoMata Local Demo Launcher script

echo "🌱 Starting GoMata Demo Environment..."

# Optional: You can uncomment these lines if mongodb and redis are not running as brew services
# echo "Starting MongoDB..."
# brew services start mongodb-community
# echo "Starting Redis..."
# brew services start redis

echo "📦 Installing backend dependencies (if any are missing)..."
(cd backend && npm install)

echo "📦 Installing frontend dependencies (if any are missing)..."
(cd frontend && npm install)

echo "🚀 Starting all microservices concurrently..."
npx -y concurrently \
  -n "BACKEND,FRONTEND,ML_SVC" \
  -c "bgBlue.bold,bgMagenta.bold,bgGreen.bold" \
  "cd backend && npm run dev" \
  "cd frontend && npm start" \
  "cd ml_service && source .venv/bin/activate && python3 main.py"
