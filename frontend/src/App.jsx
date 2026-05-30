import { useState } from 'react';
import {
  Heart,
  FlaskConical,
  Stethoscope,
  Menu,
} from 'lucide-react';
import { DiseaseProvider } from './context/DiseaseContext';
import DiseaseSelector from './components/DiseaseSelector';
import EngineeringMode from './components/EngineeringMode';
import ClinicalEmrMode from './components/ClinicalEmrMode';

const VIEWS = [
  { id: 'engineering', label: 'Engineering Mode', icon: FlaskConical, component: EngineeringMode },
  { id: 'clinical', label: 'Clinical EMR Mode', icon: Stethoscope, component: ClinicalEmrMode },
];

function AppContent() {
  const [activeView, setActiveView] = useState('engineering');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const ActiveComponent = VIEWS.find((v) => v.id === activeView)?.component || EngineeringMode;

  return (
    <div className="min-h-screen bg-clinical-bg flex">
      {/* ── Mobile sidebar overlay ── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-40 w-64 bg-white border-r border-clinical-border
          transform transition-transform duration-200 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:translate-x-0 lg:block
        `}
      >
        {/* Logo + Disease Selector */}
        <div className="h-16 flex items-center gap-3 px-6 border-b border-clinical-border">
          <div className="w-9 h-9 rounded-lg bg-primary-600 flex items-center justify-center">
            <Heart className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-gray-900 leading-tight">OmniDiag</h1>
            <p className="text-[10px] text-gray-500 leading-tight">Multi-Disease Diagnostic Platform</p>
          </div>
        </div>

        {/* Disease Selector */}
        <DiseaseSelector />

        {/* Navigation */}
        <nav className="p-4 space-y-1">
          {VIEWS.map((view) => {
            const Icon = view.icon;
            const isActive = activeView === view.id;
            return (
              <button
                key={view.id}
                onClick={() => {
                  setActiveView(view.id);
                  setSidebarOpen(false);
                }}
                className={isActive ? 'sidebar-link-active' : 'sidebar-link-inactive'}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-primary-600' : 'text-gray-400'}`} />
                {view.label}
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-clinical-border">
          <p className="text-[10px] text-gray-400 text-center">
            OmniDiag v4.0.0 &middot; Powered by XGBoost + SHAP
          </p>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <div className="flex-1 min-w-0">
        {/* Top bar (mobile) */}
        <header className="h-16 bg-white border-b border-clinical-border flex items-center justify-between px-4 lg:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>
          <div className="flex items-center gap-2">
            <Heart className="w-5 h-5 text-primary-600" />
            <span className="text-sm font-bold text-gray-900">OmniDiag</span>
          </div>
          <div className="w-9" /> {/* spacer */}
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-8 max-w-7xl mx-auto">
          <ActiveComponent />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <DiseaseProvider>
      <AppContent />
    </DiseaseProvider>
  );
}
