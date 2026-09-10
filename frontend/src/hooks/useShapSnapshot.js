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
        // Medical-term tooltips on the Y-axis are hidden via
        // invisible/opacity-0 (see ShapBarChart.jsx), which html2canvas
        // doesn't reliably honor for elements inside an SVG <foreignObject>.
        // onclone only touches the throwaway DOM clone html2canvas rasterizes
        // from — it never runs against the live page, so on-screen hover
        // behavior is untouched. display:none (unlike opacity/visibility)
        // removes the element from layout entirely, which html2canvas does
        // respect even inside foreignObject.
        onclone: (clonedDoc) => {
          clonedDoc.querySelectorAll('[role="tooltip"]').forEach((el) => {
            el.style.display = 'none'
          })
        },
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
