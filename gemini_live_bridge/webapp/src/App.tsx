import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Config from './pages/Config';
import PromptEditor from './pages/PromptEditor';
import ToolManager from './pages/ToolManager';
import Devices from './pages/Devices';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="config" element={<Config />} />
          <Route path="prompt" element={<PromptEditor />} />
          <Route path="tools" element={<ToolManager />} />
          <Route path="devices" element={<Devices />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
