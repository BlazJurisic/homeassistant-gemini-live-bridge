# Gemini Live Bridge for Home Assistant

Full-duplex voice assistant add-on using Google's Gemini Live API for natural, interruptible conversations in Croatian.

## Installation

### Add Repository to Home Assistant

1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**
2. Click the **three dots** (⋮) in the top right
3. Select **Repositories**
4. Add this URL:
   ```
   https://github.com/BlazJurisic/homeassistant-gemini-live-bridge
   ```
5. Click **Add**
6. Refresh the page
7. Find **"Gemini Live Bridge"** in the add-on store
8. Click **Install**

### Configuration

You'll need a **Google Gemini API key**. Get one from:
https://aistudio.google.com/apikey

Configure the add-on with:
- `gemini_api_key`: Your API key (required)
- `server_port`: 9872 (default)
- `log_level`: info (default)
- `croatian_personality`: true (default)
- `session_timeout_seconds`: 300 (default)

### How It Works

This add-on creates a bridge between ESP32 voice devices (like Home Assistant Voice Preview Edition) and Google's Gemini Live API, enabling:

- ✅ Full duplex conversation (can interrupt AI mid-sentence)
- ✅ Wake word activation ("Hey Jarvis")
- ✅ Croatian language support
- ✅ Natural conversation with female Croatian voice
- ✅ Direct Home Assistant device control
- ✅ Session-based operation (ends with "doviđenja" or "hvala, to je sve")

### Example Commands (Croatian)

- "Upali svjetlo u kuhinji" - Turn on kitchen light
- "Promijeni boju u plavu" - Change color to blue
- "Ugasi svjetlo" - Turn off light
- "Hvala, to je sve" - End conversation

## Usage Flow

1. Say **"Hey Jarvis"** to wake device
2. Device connects to bridge (port 9872)
3. Bridge opens Gemini Live session
4. Have natural conversation with interruptions
5. AI controls your devices via Home Assistant API
6. Say goodbye to end session

## Requirements

- Home Assistant 2025.2+ (for optimal compatibility)
- Google Gemini API key
- ESP32 voice device (custom firmware required - see documentation)

## Documentation

For complete setup and implementation details, see the full documentation in the repository.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## License

MIT License

## Credits

- Built for Home Assistant community
- Uses Google Gemini Live API
- Croatian language support
