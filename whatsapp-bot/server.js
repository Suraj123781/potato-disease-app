require('dotenv').config();
const express = require('express');
const twilio = require('twilio');
const i18n = require('i18n');
const path = require('path');

// Initialize i18n for multi-language support
i18n.configure({
  locales: ['en', 'es', 'fr', 'de', 'hi', 'ar'], // Add more languages as needed
  directory: path.join(__dirname, 'locales'),
  defaultLocale: 'en',
  cookie: 'lang',
  queryParameter: 'lang',
  autoReload: true,
  syncFiles: true
});

const app = express();
app.use(express.urlencoded({ extended: false }));
app.use(express.json());
app.use(i18n.init);

// Check for required environment variables
const requiredEnvVars = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_WHATSAPP_NUMBER'];
const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);

if (missingVars.length > 0) {
  console.error('ERROR: Missing required environment variables:', missingVars.join(', '));
  console.log('\nPlease create a .env file with the following variables:');
  console.log('TWILIO_ACCOUNT_SID=your_account_sid_here');
  console.log('TWILIO_AUTH_TOKEN=your_auth_token_here');
  console.log('TWILIO_WHATSAPP_NUMBER=+14155238886\n');
  process.exit(1);
}

// Initialize Twilio client
const client = twilio(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

// Webhook endpoint for Twilio
app.post('/webhook', async (req, res) => {
  const incomingMsg = req.body.Body.toLowerCase().trim();
  const sender = req.body.From;
  
  try {
    // Detect language (you can implement more sophisticated detection)
    let lang = 'en';
    if (incomingMsg.startsWith('es ')) {
      lang = 'es';
    } else if (incomingMsg.startsWith('fr ')) {
      lang = 'fr';
    } // Add more language detection as needed

    // Set the language for this request
    req.setLocale(lang);

    // Process the message (remove language prefix if present)
    const message = incomingMsg.replace(/^(es|fr|de|hi|ar)\s+/i, '');

    // Handle different commands
    let response = '';
    if (message === 'hello' || message === 'hi') {
      response = req.__('greeting');
    } else if (message === 'help') {
      response = req.__('help');
    } else {
      response = req.__('unknown_command');
    }

    // Send the response back to the user
    await client.messages.create({
      body: response,
      from: `whatsapp:${process.env.TWILIO_WHATSAPP_NUMBER}`,
      to: sender
    });

    res.status(200).send('Message sent');
  } catch (error) {
    console.error('Error:', error);
    res.status(500).send('Error processing message');
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
