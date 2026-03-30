import { useEffect, useState } from 'react';
import { type HADevice, getDevices } from '../api';

const TOGGLEABLE = new Set(['light', 'switch', 'fan', 'input_boolean', 'media_player']);

const DOMAIN_COLORS: Record<string, string> = {
  light: 'bg-yellow-600/20 text-yellow-400',
  switch: 'bg-green-600/20 text-green-400',
  cover: 'bg-cyan-600/20 text-cyan-400',
  climate: 'bg-orange-600/20 text-orange-400',
  fan: 'bg-teal-600/20 text-teal-400',
  media_player: 'bg-purple-600/20 text-purple-400',
  sensor: 'bg-blue-600/20 text-blue-400',
  binary_sensor: 'bg-indigo-600/20 text-indigo-400',
  input_boolean: 'bg-pink-600/20 text-pink-400',
  scene: 'bg-emerald-600/20 text-emerald-400',
  script: 'bg-amber-600/20 text-amber-400',
  automation: 'bg-rose-600/20 text-rose-400',
};

export default function Devices() {
  const [devices, setDevices] = useState<HADevice[]>([]);
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    getDevices()
      .then(setDevices)
      .catch((e) => setError(e.message));
  }, []);

  // Get unique domains
  const domains = [...new Set(devices.map((d) => d.domain))].sort();

  const filtered = devices.filter((d) => {
    if (filter !== 'all' && d.domain !== filter) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        d.entity_id.toLowerCase().includes(q) ||
        d.friendly_name.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const handleToggle = async (device: HADevice) => {
    try {
      navigator.clipboard?.writeText(device.entity_id);
    } catch { /* ignore */ }
  };

  if (error) {
    return <div className="text-red-400">{error}</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-white">Devices</h2>
        <span className="text-sm text-slate-400">{filtered.length} entities</span>
      </div>

      {/* Search + Filter */}
      <div className="flex gap-3 mb-4">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search entities..."
          className="input flex-1"
        />
      </div>

      {/* Domain tabs */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        <DomainTab
          label="All"
          active={filter === 'all'}
          onClick={() => setFilter('all')}
          count={devices.length}
        />
        {domains.map((d) => (
          <DomainTab
            key={d}
            label={d}
            active={filter === d}
            onClick={() => setFilter(d)}
            count={devices.filter((dev) => dev.domain === d).length}
          />
        ))}
      </div>

      {/* Device table */}
      <div className="border border-slate-700 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-800 text-slate-400">
            <tr>
              <th className="text-left px-4 py-2.5 font-medium">Entity</th>
              <th className="text-left px-4 py-2.5 font-medium">Name</th>
              <th className="text-left px-4 py-2.5 font-medium">State</th>
              <th className="text-left px-4 py-2.5 font-medium">Domain</th>
              <th className="px-4 py-2.5 font-medium w-16"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {filtered.map((d) => (
              <tr key={d.entity_id} className="hover:bg-slate-800/50">
                <td className="px-4 py-2.5 font-mono text-xs text-slate-300">
                  {d.entity_id}
                </td>
                <td className="px-4 py-2.5 text-white">{d.friendly_name}</td>
                <td className="px-4 py-2.5">
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                      d.state === 'on'
                        ? 'bg-green-600/20 text-green-400'
                        : d.state === 'off'
                        ? 'bg-slate-700 text-slate-400'
                        : 'bg-slate-700 text-slate-300'
                    }`}
                  >
                    {d.state}
                  </span>
                </td>
                <td className="px-4 py-2.5">
                  <span className={`inline-block px-2 py-0.5 rounded text-xs ${DOMAIN_COLORS[d.domain] || 'bg-slate-700 text-slate-400'}`}>
                    {d.domain}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-center">
                  {TOGGLEABLE.has(d.domain) && (
                    <button
                      onClick={() => handleToggle(d)}
                      title="Copy entity_id"
                      className="text-slate-500 hover:text-white transition-colors text-xs"
                    >
                      copy
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                  No devices found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DomainTab({
  label,
  active,
  onClick,
  count,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  count: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 rounded-full text-xs transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
      }`}
    >
      {label} ({count})
    </button>
  );
}
