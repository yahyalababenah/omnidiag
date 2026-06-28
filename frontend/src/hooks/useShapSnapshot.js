/**
 * useShapSnapshot — capture a DOM node as a PNG data-URL for PDF embedding.
 *
 * Usage:
 *   const { snapshot, capture } = useShapSnapshot()
 *   <div ref={chartRef}>...</div>
 *   <button onClick={() => capture(chartRef.current)}>Export PDF</button>
 *   // then pass `snapshot` to PDFReport as shapImageUrl
 */
import { useState, useCallback } from 'react'

export function useShapSnapshot() {
  const [snapshot, setSnapshot] = useState(null)
  const [capturing, setCapturing] = useState(false)

  const capture = useCallback(async (element) => {
    if (!element) return null
    setCapturing(true)
    try {
      const { default: html2canvas } = await import('html2canvas')
      const canvas = await html2canvas(element, {
        backgroundColor: '#ffffff',
        scale: 2,           // 2× DPI for crisp PDF embedding
        useCORS: true,
        logging: false,
      })
      const dataUrl = canvas.toDataURL('image/png')
      setSnapshot(dataUrl)
      return dataUrl
    } catch (err) {
      console.warn('useShapSnapshot: capture failed', err)
      return null
    } finally {
      setCapturing(false)
    }
  }, [])

  return { snapshot, capturing, capture }
}
