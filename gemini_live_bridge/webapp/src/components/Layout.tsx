import { NavLink, Outlet } from 'react-router-dom';

const links = [
  { to: '/', label: 'Dashboard', icon: '⊞' },
  { to: '/config', label: 'Config', icon: '⚙' },
  { to: '/prompt', label: 'Prompt', icon: '✎' },
  { to: '/tools', label: 'Tools', icon: '⚡' },
  { to: '/devices', label: 'Devices', icon: '◉' },
];

export default function Layout() {
  return (
    <div className="flex h-screen bg-slate-900">
      {/* Sidebar */}
      <nav className="w-56 shrink-0 border-r border-slate-700 bg-slate-800 flex flex-col">
        <div className="p-4 border-b border-slate-700">
          <h1 className="text-lg font-semibold text-white">Voice Bridge</h1>
          <p className="text-xs text-slate-400 mt-0.5">Dashboard</p>
        </div>
        <ul className="flex-1 py-2">
          {links.map((l) => (
            <li key={l.to}>
              <NavLink
                to={l.to}
                end={l.to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                    isActive
                      ? 'bg-blue-600/20 text-blue-400 border-r-2 border-blue-400'
                      : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                  }`
                }
              >
                <span className="text-base">{l.icon}</span>
                {l.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
