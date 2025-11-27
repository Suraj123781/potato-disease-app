# WhatsApp Bot with Twilio

A 24/7 WhatsApp bot with multi-language support built with Node.js and Twilio.

## Features

- Multi-language support (English, Spanish, and more)
- 24/7 availability when deployed
- Easy to extend with new commands
- Environment-based configuration

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Twilio account with WhatsApp Sandbox access
- ngrok (for local development)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and fill in your Twilio credentials:
   ```
   cp .env.example .env
   ```
4. Update the `.env` file with your Twilio credentials

## Running Locally

1. Start the server:
   ```bash
   npm start
   ```
2. Use ngrok to expose your local server:
   ```bash
   ngrok http 3000
   ```
3. Configure your Twilio Sandbox to point to your ngrok URL (e.g., `https://your-ngrok-url.ngrok.io/webhook`)

## Deploying to Production

For 24/7 availability, deploy to a cloud provider like:
- Heroku
- AWS
- Google Cloud
- Microsoft Azure

## Adding New Languages

1. Add the language code to the `locales` array in `server.js`
2. Create a new JSON file in the `locales` directory (e.g., `fr.json` for French)
3. Add translations for all message keys

## Available Commands

- `hello`/`hi` - Get a greeting
- `help` - Show help message

## License

MIT
