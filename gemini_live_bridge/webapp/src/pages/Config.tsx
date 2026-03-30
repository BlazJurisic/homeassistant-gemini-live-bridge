import { useEffect, useState } from 'react';
import { type BridgeConfig, getConfig, saveConfig } from '../api';

const PROVIDERS = ['gemini', 'openai', 'hybrid'];
const GEMINI_VOICES = ['Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'];
const OPENAI_VOICES = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse'];
const LOG_LEVELS = ['debug', 'info', 'warning', 'error'];

export default function Config() {
  const [config, setConfig] = useState<BridgeConfig>({});
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch((e) => setError(e.message));
  }, []);

  const update = (key: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    setMessage('');
    try {
      await saveConfig(config);
      setMessage('Saved! Restart the add-on to apply changes.');
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    }
    setSaving(false);
  };

  if (error) {
    return <div className="text-red-400">{error}</div>;
  }

  return (
    <div className="max-w-2xl">
      <h2 className="text-2xl font-semibold text-white mb-6">Configuration</h2>

      <div className="space-y-5">
        {/* Provider */}
        <Field label="Provider">
          <select
            value={config.provider || 'gemini'}
            onChange={(e) => update('provider', e.target.value)}
            className="input"
          >
            {PROVIDERS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </Field>

        {/* Gemini settings */}
        {(config.provider === 'gemini' || !config.provider) && (
          <>
            <Field label="Gemini API Key">
              <input
                type="password"
                value={config.gemini_api_key || ''}
                onChange={(e) => update('gemini_api_key', e.target.value)}
                placeholder="AIza..."
                className="input"
              />
            </Field>
            <Field label="Gemini Voice">
              <select
                value={config.gemini_voice || 'Zephyr'}
                onChange={(e) => update('gemini_voice', e.target.value)}
                className="input"
              >
                {GEMINI_VOICES.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </Field>
          </>
        )}

        {/* OpenAI settings */}
        {config.provider === 'openai' && (
          <>
            <Field label="OpenAI API Key">
              <input
                type="password"
                value={config.openai_api_key || ''}
                onChange={(e) => update('openai_api_key', e.target.value)}
                placeholder="sk-..."
                className="input"
              />
            </Field>
            <Field label="OpenAI Voice">
              <select
                value={config.openai_voice || 'alloy'}
                onChange={(e) => update('openai_voice', e.target.value)}
                className="input"
              >
                {OPENAI_VOICES.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </Field>
          </>
        )}

        {/* Common settings */}
        <Field label="Log Level">
          <select
            value={config.log_level || 'info'}
            onChange={(e) => update('log_level', e.target.value)}
            className="input"
          >
            {LOG_LEVELS.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </Field>

        <Field label="Session Timeout (seconds)">
          <input
            type="number"
            value={config.session_timeout_seconds || 300}
            onChange={(e) => update('session_timeout_seconds', Number(e.target.value))}
            className="input"
          />
        </Field>

        <Field label="Croatian Personality">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.croatian_personality !== false}
              onChange={(e) => update('croatian_personality', e.target.checked)}
              className="w-4 h-4 rounded border-slate-600 bg-slate-700"
            />
            <span className="text-sm text-slate-300">Enable Croatian language personality</span>
          </label>
        </Field>

        {/* Save */}
        <div className="flex items-center gap-4 pt-2">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors text-sm font-medium"
          >
            {saving ? 'Saving...' : 'Save Configuration'}
          </button>
          {message && (
            <span className={`text-sm ${message.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
              {message}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-1.5">{label}</label>
      {children}
    </div>
  );
}
