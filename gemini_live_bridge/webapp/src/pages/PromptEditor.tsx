import { useEffect, useRef, useState } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import { getPrompt, savePrompt } from '../api';

const DEFAULT_PROMPT = `Pozdrav! Ja sam Jarvis, tvoj virtualni asistent u pametnoj kući.

VAŽNO:
- Govoriš isključivo hrvatski jezik
- Muškog si roda (ja sam Jarvis)
- Prijateljski i prirodan ton razgovora
- Kratki i jasni odgovori (ne duljiti)
- Odgovaram na SVA pitanja

MOGUĆNOSTI:
- Kontrola pametne kuće (rasvjeta, uređaji)
- Odgovaranje na bilo koja pitanja
- Razgovori o bilo kojoj temi

PRAVILA RAZGOVORA:
- Kada korisnik kaže "hvala", "to je sve", "doviđenja" - pozovi end_conversation()
- Potvrdi svaku izvršenu akciju kratko i jasno
- Uvijek pokušaj pomoći, bez obzira na temu pitanja`;

export default function PromptEditor() {
  const [prompt, setPrompt] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  useEffect(() => {
    getPrompt()
      .then(({ prompt: p }) => {
        setPrompt(p || '');
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  const handleSave = async () => {
    const value = editorRef.current?.getValue() || '';
    setSaving(true);
    setMessage('');
    try {
      await savePrompt(value);
      setMessage('Saved! New sessions will use this prompt.');
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    }
    setSaving(false);
  };

  const handleReset = () => {
    editorRef.current?.setValue(DEFAULT_PROMPT);
  };

  if (!loaded) return <div className="text-slate-400">Loading...</div>;

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-semibold text-white">System Prompt</h2>
          <p className="text-sm text-slate-400 mt-1">
            Edit the system prompt used for voice sessions. Leave empty to use the default.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleReset}
            className="px-4 py-2 text-sm bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
          >
            Reset to Default
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-5 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors font-medium"
          >
            {saving ? 'Saving...' : 'Save Prompt'}
          </button>
        </div>
      </div>
      {message && (
        <div className={`text-sm mb-3 ${message.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
          {message}
        </div>
      )}
      <div className="flex-1 border border-slate-700 rounded-lg overflow-hidden">
        <Editor
          defaultLanguage="plaintext"
          defaultValue={prompt || DEFAULT_PROMPT}
          theme="vs-dark"
          onMount={(editor) => { editorRef.current = editor; }}
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            lineNumbers: 'on',
            wordWrap: 'on',
            padding: { top: 12 },
          }}
        />
      </div>
    </div>
  );
}
