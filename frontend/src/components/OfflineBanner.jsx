import { useEffect, useState } from 'react'
import { WifiOff, X } from 'lucide-react'

/**
 * Displays a dismissible banner when the browser loses network connectivity.
 * Automatically hides when connectivity is restored.
 */
export default function OfflineBanner() {
  const [offline, setOffline] = useState(!navigator.onLine)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const goOffline = () => { setOffline(true); setDismissed(false) }
    const goOnline  = () => { setOffline(false) }
    window.addEventListener('offline', goOffline)
    window.addEventListener('online',  goOnline)
    return () => {
      window.removeEventListener('offline', goOffline)
      window.removeEventListener('online',  goOnline)
    }
  }, [])

  if (!offline || dismissed) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between gap-3 bg-amber-600 px-4 py-2.5 text-white shadow-lg">
      <div className="flex items-center gap-2 text-sm font-medium">
        <WifiOff size={16} className="shrink-0" />
        <span>
          You are offline. Cached results are still available — new predictions
          require a network connection.
        </span>
      </div>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss offline banner"
        className="rounded p-1 hover:bg-amber-700 transition-colors shrink-0"
      >
        <X size={16} />
      </button>
    </div>
  )
}
