/**
 * OmniDiag — Clinical PDF Report
 *
 * Rendered entirely client-side via @react-pdf/renderer.
 * No server round-trip; generates in the browser and downloads immediately.
 *
 * Layout:
 *   1. Header  — OmniDiag logo text + report date
 *   2. Patient — name, DOB, gender, MRN (if available)
 *   3. Diagnosis — prediction badge, confidence bar, clinical summary
 *   4. SHAP chart — embedded as a PNG snapshot (passed as dataUrl prop)
 *   5. Counterfactuals — up to 3 scenarios as a formatted table
 *   6. Footer — doctor name, clinic, signature line
 */

import {
  Document,
  Page,
  Text,
  View,
  StyleSheet,
  Image,
  Font,
} from '@react-pdf/renderer'

// ── Colour palette (matches Tailwind clinical theme) ──────────────────────────
const C = {
  primary:    '#0f172a',   // slate-900
  accent:     '#2563eb',   // blue-600
  positive:   '#dc2626',   // red-600 — high risk
  negative:   '#16a34a',   // green-600 — low risk
  border:     '#e2e8f0',   // slate-200
  muted:      '#64748b',   // slate-500
  light:      '#f8fafc',   // slate-50
  white:      '#ffffff',
}

const styles = StyleSheet.create({
  page: {
    fontFamily: 'Helvetica',
    fontSize: 9,
    color: C.primary,
    paddingTop: 40,
    paddingBottom: 60,
    paddingHorizontal: 48,
    backgroundColor: C.white,
  },

  // ── Header ─────────────────────────────────────────────────────────────────
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    borderBottomWidth: 2,
    borderBottomColor: C.accent,
    paddingBottom: 10,
    marginBottom: 16,
  },
  logoText: { fontSize: 18, fontFamily: 'Helvetica-Bold', color: C.accent },
  logoSub:  { fontSize: 8,  color: C.muted, marginTop: 2 },
  headerRight: { alignItems: 'flex-end' },
  headerLabel: { fontSize: 7, color: C.muted, textTransform: 'uppercase', letterSpacing: 0.5 },
  headerValue: { fontSize: 9, color: C.primary, marginTop: 1 },

  // ── Section ────────────────────────────────────────────────────────────────
  section: { marginBottom: 14 },
  sectionTitle: {
    fontSize: 10,
    fontFamily: 'Helvetica-Bold',
    color: C.accent,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    paddingBottom: 4,
    marginBottom: 8,
  },

  // ── 2-column row ───────────────────────────────────────────────────────────
  row: { flexDirection: 'row', marginBottom: 4 },
  cell: { flex: 1 },
  label: { fontSize: 7, color: C.muted, textTransform: 'uppercase', letterSpacing: 0.4 },
  value: { fontSize: 9, color: C.primary, marginTop: 1 },

  // ── Diagnosis badge ────────────────────────────────────────────────────────
  badgeRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 10 },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
    fontSize: 11,
    fontFamily: 'Helvetica-Bold',
  },

  // ── Risk probability bar ──────────────────────────────────────────────────
  confLabel: { fontSize: 7, color: C.muted, marginBottom: 3 },
  confBarBg: { height: 6, backgroundColor: C.border, borderRadius: 3, width: 160 },
  confBarFill: { height: 6, borderRadius: 3 },

  // ── Clinical summary text ──────────────────────────────────────────────────
  summaryBox: {
    backgroundColor: C.light,
    borderLeftWidth: 3,
    borderLeftColor: C.accent,
    padding: 8,
    borderRadius: 2,
    marginTop: 6,
  },
  summaryText: { fontSize: 9, color: C.primary, lineHeight: 1.5 },

  // ── SHAP chart image ───────────────────────────────────────────────────────
  chartImage: { width: '100%', maxHeight: 180, objectFit: 'contain', marginTop: 4 },

  // ── Counterfactuals table ──────────────────────────────────────────────────
  table: { borderWidth: 1, borderColor: C.border, borderRadius: 2 },
  tableHeader: {
    flexDirection: 'row',
    backgroundColor: C.light,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    paddingVertical: 5,
    paddingHorizontal: 8,
  },
  tableHeaderCell: { fontFamily: 'Helvetica-Bold', fontSize: 7, color: C.muted, textTransform: 'uppercase' },
  tableRow: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    paddingVertical: 5,
    paddingHorizontal: 8,
  },
  tableCell: { fontSize: 8, color: C.primary },
  feasBadge: {
    fontSize: 7,
    fontFamily: 'Helvetica-Bold',
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 3,
  },

  // ── Footer ─────────────────────────────────────────────────────────────────
  footer: {
    position: 'absolute',
    bottom: 30,
    left: 48,
    right: 48,
    borderTopWidth: 1,
    borderTopColor: C.border,
    paddingTop: 8,
  },
  footerRow: { flexDirection: 'row', justifyContent: 'space-between' },
  footerLabel: { fontSize: 7, color: C.muted },
  footerValue: { fontSize: 8, color: C.primary },
  signatureLine: {
    borderBottomWidth: 1,
    borderBottomColor: C.primary,
    width: 160,
    marginTop: 20,
    marginBottom: 2,
  },
  signatureLabel: { fontSize: 7, color: C.muted },
  pageNum: { fontSize: 7, color: C.muted, textAlign: 'center', marginTop: 4 },
})

// ── Helpers ───────────────────────────────────────────────────────────────────

function LabelValue({ label, value }) {
  return (
    <View style={styles.cell}>
      <Text style={styles.label}>{label}</Text>
      <Text style={styles.value}>{value || '—'}</Text>
    </View>
  )
}

function feasColor(f) {
  if (f === 'high')   return { backgroundColor: '#dcfce7', color: '#15803d' }
  if (f === 'medium') return { backgroundColor: '#fef9c3', color: '#a16207' }
  return                       { backgroundColor: '#fee2e2', color: '#b91c1c' }
}

// ── Main PDF document ─────────────────────────────────────────────────────────

export default function PDFReport({
  patient,
  disease,
  result,
  shapText,
  counterfactuals,
  shapImageUrl,
  doctorName,
  clinicName,
  reportDate,
}) {
  const isPositive = result?.prediction === 1
  const confidence = Math.round((result?.confidence ?? 0) * 100)
  const diagnosisLabel = result?.diagnosis ?? (isPositive ? 'Positive' : 'Negative')
  const badgeStyle = isPositive
    ? { ...styles.badge, backgroundColor: '#fee2e2', color: C.positive }
    : { ...styles.badge, backgroundColor: '#dcfce7', color: C.negative }
  const barColor = isPositive ? C.positive : C.negative

  const cfs = (counterfactuals ?? []).slice(0, 3)
  const dateStr = reportDate ?? new Date().toLocaleDateString('en-GB')

  return (
    <Document
      title={`OmniDiag Report — ${patient?.full_name ?? 'Patient'}`}
      author="OmniDiag Clinical Platform"
    >
      <Page size="A4" style={styles.page}>

        {/* ── Header ── */}
        <View style={styles.header}>
          <View>
            <Text style={styles.logoText}>OmniDiag</Text>
            <Text style={styles.logoSub}>Multi-Disease Clinical Decision Support</Text>
          </View>
          <View style={styles.headerRight}>
            <Text style={styles.headerLabel}>Report Date</Text>
            <Text style={styles.headerValue}>{dateStr}</Text>
            <Text style={[styles.headerLabel, { marginTop: 6 }]}>Disease Module</Text>
            <Text style={styles.headerValue}>
              {disease?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) ?? '—'}
            </Text>
          </View>
        </View>

        {/* ── Patient Information ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Patient Information</Text>
          <View style={styles.row}>
            <LabelValue label="Full Name"    value={patient?.full_name} />
            <LabelValue label="MRN"          value={patient?.mrn} />
          </View>
          <View style={styles.row}>
            <LabelValue label="Date of Birth" value={patient?.date_of_birth} />
            <LabelValue label="Gender"        value={patient?.gender} />
          </View>
          {patient?.contact_email && (
            <View style={styles.row}>
              <LabelValue label="Contact Email" value={patient.contact_email} />
              <View style={styles.cell} />
            </View>
          )}
        </View>

        {/* ── Diagnosis Summary ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>AI Diagnosis Summary</Text>
          <View style={styles.badgeRow}>
            <Text style={badgeStyle}>{diagnosisLabel}</Text>
            <View>
              <Text style={styles.confLabel}>Risk Probability</Text>
              <View style={styles.confBarBg}>
                <View style={[styles.confBarFill, { width: `${confidence}%`, backgroundColor: barColor }]} />
              </View>
              <Text style={[styles.confLabel, { marginTop: 2 }]}>{confidence}%</Text>
            </View>
          </View>
          {shapText && (
            <View style={styles.summaryBox}>
              <Text style={styles.summaryText}>{shapText}</Text>
            </View>
          )}
        </View>

        {/* ── SHAP Chart ── */}
        {shapImageUrl && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Feature Impact Analysis (SHAP)</Text>
            <Image src={shapImageUrl} style={styles.chartImage} />
          </View>
        )}

        {/* ── Recommended Interventions ── */}
        {cfs.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Recommended Lifestyle Interventions</Text>
            <View style={styles.table}>
              <View style={styles.tableHeader}>
                <Text style={[styles.tableHeaderCell, { flex: 3 }]}>Scenario</Text>
                <Text style={[styles.tableHeaderCell, { flex: 1 }]}>Risk Reduction</Text>
                <Text style={[styles.tableHeaderCell, { flex: 1 }]}>Feasibility</Text>
              </View>
              {cfs.map((cf, i) => (
                <View key={i} style={[styles.tableRow, i === cfs.length - 1 && { borderBottomWidth: 0 }]}>
                  <Text style={[styles.tableCell, { flex: 3 }]}>{cf.scenario}</Text>
                  <Text style={[styles.tableCell, { flex: 1 }]}>{cf.risk_reduction}</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.feasBadge, feasColor(cf.feasibility)]}>
                      {cf.feasibility?.toUpperCase()}
                    </Text>
                  </View>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* ── Footer ── */}
        <View style={styles.footer} fixed>
          <View style={styles.footerRow}>
            <View>
              <View style={styles.signatureLine} />
              <Text style={styles.signatureLabel}>
                Dr. {doctorName ?? 'Physician Name'} — {clinicName ?? 'OmniDiag Clinic'}
              </Text>
            </View>
            <View style={{ alignItems: 'flex-end' }}>
              <Text style={styles.footerLabel}>Generated by</Text>
              <Text style={styles.footerValue}>OmniDiag v4.0.0</Text>
              <Text style={[styles.footerLabel, { marginTop: 4 }]}>
                For clinical decision support only. Not a substitute for professional medical judgment.
              </Text>
            </View>
          </View>
          <Text style={styles.pageNum} render={({ pageNumber, totalPages }) =>
            `Page ${pageNumber} of ${totalPages}`
          } />
        </View>

      </Page>
    </Document>
  )
}
