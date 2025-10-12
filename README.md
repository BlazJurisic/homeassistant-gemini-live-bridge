# Gemini Live Bridge Add-on

Full-duplex voice assistant bridge for Home Assistant using Google's Gemini Live API.

## Configuration

### Required:
- **gemini_api_key**: Your Google Gemini API key (get it from https://aistudio.google.com/apikey)

### Optional:
- **server_port**: TCP port for device connections (default: 9872)
- **log_level**: Logging level - debug, info, warning, error (default: info)
- **croatian_personality**: Use Croatian personality for the assistant (default: true)
- **session_timeout_seconds**: Automatic session timeout (default: 300)

## How it Works

This add-on creates a bridge between ESP32 voice devices and Google's Gemini Live API, enabling full-duplex conversations with Home Assistant device control.

### Flow:
1. Voice device detects wake word ("Hey Jarvis")
2. Device connects to this bridge (port 9872)
3. Bridge opens Gemini Live session
4. Audio streams bidirectionally
5. Gemini can call Home Assistant services to control devices
6. Session ends when user says goodbye or after timeout

## First Time Setup

1. Get your Gemini API key from https://aistudio.google.com/apikey
2. Install this add-on
3. Configure your API key in the add-on configuration
4. Start the add-on
5. Check logs to verify it's running

## Usage

The bridge listens on port 9872 for incoming connections from voice devices.

### Supported Commands (Croatian):
- "Upali svjetlo u kuhinji" - Turn on kitchen light
- "Promijeni boju u plavu" - Change color to blue
- "Ugasi svjetlo" - Turn off light
- "Hvala, to je sve" - End conversation

## Troubleshooting

### Check Logs:
Go to Add-on → Log tab to see what's happening

### Common Issues:

**"GEMINI_API_KEY not set!"**
- Add your API key in the add-on configuration

**"Failed to connect to Home Assistant"**
- The add-on should automatically connect via Supervisor
- Check that Home Assistant is running

**"No device connections"**
- Ensure your ESP32 device is configured with correct bridge host
- Check network connectivity
- Verify port 9872 is accessible

## Function Calling

The bridge exposes these functions to Gemini:

1. **control_device**: Turn on/off lights, change colors, set brightness
2. **query_device_state**: Get current state of devices
3. **end_conversation**: End the session and return to wake word mode

## Support

For issues and documentation, see:
- Main documentation: `/config/TODO_FULL_DUPLEX.md`
- Environment context: `/config/claude.md`
