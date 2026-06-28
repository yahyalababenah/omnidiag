{{- define "omnidiag.fullname" -}}
{{- printf "omnidiag" -}}
{{- end }}

{{- define "omnidiag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" -}}
{{- end }}
