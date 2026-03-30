import { useEffect, useState } from 'react';
import { type BridgeStatus, getStatus } from '../api';

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}h ${m}m ${s}s`;
}

export default function Dashboard() {
  const [status, setStatus] = useState<BridgeStatus | null>(null);
  const [error, setError] = useState('');

  const refresh = () => {
    getStatus().then(setStatus).catch((e) => setError(e.message));
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  if (error) {
    return (
      <div className="text-red-400 bg-red-900/20 p-4 rounded-lg">
        Failed to connect: {error}
      </div>
    );
  }

  if (!status) {
    return <div className="text-slate-400">Loading...</div>;
  }

  const cards = [
    {
      label: 'Provider',
      value: status.provider.toUpperCase(),
      color: 'text-blue-400',
    },
    {
      label: 'Uptime',
      value: formatUptime(status.uptime_seconds),
      color: 'text-green-400',
    },
    {
      label: 'HA Connected',
      value: status.ha_connected ? 'Yes' : 'No',
      color: status.ha_connected ? 'text-green-400' : 'text-red-400',
    },
    {
      label: 'Tools Loaded',
      value: String(status.tools_loaded),
      color: 'text-purple-400',
    },
  ];

  return (
    <div>
      <h2 className="text-2xl font-semibold text-white mb-6">Dashboard</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((c) => (
          <div
            key={c.label}
            className="bg-slate-800 border border-slate-700 rounded-lg p-5"
          >
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">
              {c.label}
            </p>
            <p className={`text-2xl font-bold ${c.color}`}>{c.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
