import { useEffect, useRef, useState } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import {
  type ToolSummary,
  getTools,
  getTool,
  createTool,
  updateTool,
  deleteTool,
} from '../api';

const TEMPLATE = `from tools.base import BaseTool


class CustomTool(BaseTool):
    @property
    def name(self):
        return "my_tool"

    @property
    def description(self):
        return "Description for the LLM"

    @property
    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "A parameter"}
            },
            "required": ["param1"]
        }

    async def execute(self, args, ha_client):
        # Your logic here
        return {"result": "ok"}


tool = CustomTool()
`;

export default function ToolManager() {
  const [tools, setTools] = useState<ToolSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [code, setCode] = useState('');
  const [isBuiltin, setIsBuiltin] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [message, setMessage] = useState('');
  const [saving, setSaving] = useState(false);
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  const refreshTools = () => {
    getTools().then(setTools).catch(() => {});
  };

  useEffect(() => {
    refreshTools();
  }, []);

  const selectTool = async (name: string) => {
    setCreating(false);
    setMessage('');
    try {
      const detail = await getTool(name);
      setSelected(name);
      setCode(detail.code || '# Builtin tool — read-only');
      setIsBuiltin(detail.builtin);
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    }
  };

  const handleCreate = () => {
    setCreating(true);
    setSelected(null);
    setNewName('');
    setCode(TEMPLATE);
    setIsBuiltin(false);
    setMessage('');
  };

  const handleSave = async () => {
    const value = editorRef.current?.getValue() || '';
    setSaving(true);
    setMessage('');
    try {
      if (creating) {
        if (!newName.trim()) {
          setMessage('Enter a tool name');
          setSaving(false);
          return;
        }
        await createTool(newName.trim(), value);
        setCreating(false);
        setSelected(newName.trim());
      } else if (selected) {
        await updateTool(selected, value);
      }
      refreshTools();
      setMessage('Saved! Restart the add-on to reload tools.');
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    }
    setSaving(false);
  };

  const handleDelete = async () => {
    if (!selected || isBuiltin) return;
    if (!confirm(`Delete tool "${selected}"?`)) return;
    try {
      await deleteTool(selected);
      setSelected(null);
      setCode('');
      refreshTools();
      setMessage('Tool deleted.');
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    }
  };

  return (
    <div className="flex h-[calc(100vh-3rem)] gap-4">
      {/* Tool list */}
      <div className="w-60 shrink-0 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-semibold text-white">Tools</h2>
          <button
            onClick={handleCreate}
            className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            + New
          </button>
        </div>
        <div className="flex-1 overflow-auto space-y-1">
          {tools.map((t) => (
            <button
              key={t.name}
              onClick={() => selectTool(t.name)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                selected === t.name
                  ? 'bg-blue-600/20 text-blue-400'
                  : 'text-slate-300 hover:bg-slate-800'
              }`}
            >
              <div className="font-medium">{t.name}</div>
              <div className="text-xs text-slate-500 truncate">
                {t.builtin ? '(builtin) ' : ''}{t.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Editor area */}
      <div className="flex-1 flex flex-col">
        {(selected || creating) ? (
          <>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                {creating ? (
                  <input
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="tool_name"
                    className="input text-sm w-48"
                  />
                ) : (
                  <h3 className="text-lg font-medium text-white">{selected}</h3>
                )}
                {isBuiltin && (
                  <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded">
                    read-only
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                {!isBuiltin && selected && (
                  <button
                    onClick={handleDelete}
                    className="px-3 py-1.5 text-sm bg-red-600/20 text-red-400 rounded-lg hover:bg-red-600/30 transition-colors"
                  >
                    Delete
                  </button>
                )}
                {!isBuiltin && (
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors font-medium"
                  >
                    {saving ? 'Saving...' : 'Save'}
                  </button>
                )}
              </div>
            </div>
            {message && (
              <div className={`text-sm mb-2 ${message.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
                {message}
              </div>
            )}
            <div className="flex-1 border border-slate-700 rounded-lg overflow-hidden">
              <Editor
                key={selected || 'new'}
                defaultLanguage="python"
                defaultValue={code}
                theme="vs-dark"
                onMount={(editor) => { editorRef.current = editor; }}
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  readOnly: isBuiltin,
                  padding: { top: 12 },
                }}
              />
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500">
            Select a tool or create a new one
          </div>
        )}
      </div>
    </div>
  );
}
