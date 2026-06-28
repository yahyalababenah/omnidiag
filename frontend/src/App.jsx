import { useState } from 'react';
import {
  Heart,
  FlaskConical,
  Stethoscope,
  Shield,
  Table2,
  ArrowLeftRight,
  LogIn,
  LogOut,
  Moon,
  Sun,
  Menu,
} from 'lucide-react';
import { DiseaseProvider } from './context/DiseaseContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import DiseaseSelector from './components/DiseaseSelector';
import EngineeringMode from './components/EngineeringMode';
import ClinicalEmrMode from './components/ClinicalEmrMode';
import AdminDashboard from './components/AdminDashboard';
import BatchUpload from './components/BatchUpload';
import ComparisonMode from './components/ComparisonMode';
import OfflineBanner from './components/OfflineBanner';
import InstallPrompt from './components/InstallPrompt';
import LoginModal from './components/LoginModal';

const CLINICAL_VIEWS = [
  { id: 'engineering', label: 'Engineering Mode',    icon: FlaskConical,   component: EngineeringMode },
  { id: 'clinical',    label: 'Clinical EMR Mode',   icon: Stethoscope,    component: ClinicalEmrMode },
  { id: 'batch',       label: 'Batch Prediction',    icon: Table2,         component: BatchUpload },
  { id: 'comparison',  label: 'Before/After Compare', icon: ArrowLeftRight, component: ComparisonMode },
];

function AppContent() {
  const [activeView, setActiveView]   = useState('engineering');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showLogin, setShowLogin]     = useState(false);
  const [pendingView, setPendingView] = useState(null);
  const { user, isAdmin, logout }     = useAuth();
  const { dark, toggle: toggleTheme } = useTheme();

  const isAdminView = activeView === 'admin';

  let ActiveComponent = CLINICAL_VIEWS.find((v) => v.id === activeView)?.component || EngineeringMode;
  if (isAdminView) ActiveComponent = AdminDashboard;

  function handleAdminClick() {
    if (user) {
      setActiveView('admin');
      setSidebarOpen(false);
    } else {
      setPendingView('admin');
      setShowLogin(true);
    }
  }

  function handleLoginSuccess() {
    if (pendingView) {
      setActiveView(pendingView);
      setPendingView(null);
    }
  }

  return (
    <div className="min-h-screen bg-clinical-bg flex dark:bg-slate-950">
      {/* Skip to main content — WCAG 2.1 SC 2.4.1 */}
      <a href="#main-content" className="skip-link">Skip to main content</a>

      <OfflineBanner />

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        aria-label="Navigation"
        className={`
          fixed lg:static inset-y-0 left-0 z-40 w-64 bg-white border-r border-clinical-border
          dark:bg-slate-900 dark:border-slate-700
          transform transition-transform duration-200 ease-in-out flex flex-col
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:translate-x-0 lg:flex
        `}
      >
        {/* Logo */}
        <div className="h-16 flex items-center gap-3 px-6 border-b border-clinical-border flex-shrink-0">
          <div className="w-9 h-9 rounded-lg bg-primary-600 flex items-center justify-center">
            <Heart className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-gray-900 leading-tight">OmniDiag</h1>
            <p className="text-[10px] text-gray-500 leading-tight">Multi-Disease Diagnostic Platform</p>
          </div>
        </div>

        {/* Disease selector — only relevant for clinical views */}
        {!isAdminView && <DiseaseSelector />}

        {/* Navigation */}
        <nav className="p-4 space-y-1 flex-1 overflow-y-auto">
          {CLINICAL_VIEWS.map((view) => {
            const Icon = view.icon;
            const isActive = activeView === view.id;
            return (
              <button
                key={view.id}
                onClick={() => { setActiveView(view.id); setSidebarOpen(false); }}
                className={isActive ? 'sidebar-link-active w-full text-left' : 'sidebar-link-inactive w-full text-left'}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-primary-600' : 'text-gray-400'}`} />
                {view.label}
              </button>
            );
          })}

          {/* Admin section divider */}
          <div className="pt-2 mt-2 border-t border-clinical-border">
            <button
              onClick={handleAdminClick}
              className={activeView === 'admin' ? 'sidebar-link-active w-full text-left' : 'sidebar-link-inactive w-full text-left'}
            >
              <Shield className={`w-5 h-5 ${activeView === 'admin' ? 'text-primary-600' : 'text-gray-400'}`} />
              Admin Dashboard
              {!user && <span className="ml-auto text-[10px] text-gray-400">Sign in</span>}
            </button>
          </div>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-clinical-border space-y-2 flex-shrink-0">
          {user ? (
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <p className="text-xs font-medium text-gray-900 truncate">{user.full_name}</p>
                <p className="text-[10px] text-gray-500 truncate">{user.email}</p>
              </div>
              <button
                onClick={logout}
                title="Sign out"
                className="p-1.5 rounded-lg hover:bg-red-50 text-gray-400 hover:text-red-600 transition-colors flex-shrink-0"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowLogin(true)}
              className="btn-secondary w-full text-xs flex items-center gap-2"
            >
              <LogIn className="w-4 h-4" /> Sign In
            </button>
          )}
          {/* Dark mode toggle */}
          <button
            onClick={toggleTheme}
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="btn-secondary w-full text-xs flex items-center gap-2"
          >
            {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            {dark ? 'Light Mode' : 'Dark Mode'}
          </button>
          <InstallPrompt />
          <p className="text-[10px] text-gray-400 text-center dark:text-slate-600">
            OmniDiag v4.0.0 &middot; Powered by XGBoost + SHAP
          </p>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <div className="flex-1 min-w-0">
        {/* Mobile top bar */}
        <header className="h-16 bg-white border-b border-clinical-border flex items-center justify-between px-4 lg:hidden dark:bg-slate-900 dark:border-slate-700">
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
          <div className="flex items-center gap-2">
            <InstallPrompt />
          </div>
        </header>

        {/* Page content */}
        <main id="main-content" className="p-4 lg:p-8 max-w-7xl mx-auto" role="main">
          {isAdminView && !isAdmin ? (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <Shield className="w-12 h-12 text-gray-300 mb-4" />
              <p className="text-gray-500 text-sm">Admin access required.</p>
              <button onClick={() => setShowLogin(true)} className="btn-primary mt-4 text-sm">
                Sign In as Admin
              </button>
            </div>
          ) : (
            <ActiveComponent />
          )}
        </main>
      </div>

      {showLogin && <LoginModal onClose={() => setShowLogin(false)} onSuccess={handleLoginSuccess} />}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <DiseaseProvider>
          <AppContent />
        </DiseaseProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
