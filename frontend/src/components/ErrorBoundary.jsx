/**
 * ErrorBoundary.jsx
 * ==================
 * React Error Boundary component that catches render-phase errors
 * and displays a fallback UI instead of crashing the entire app.
 *
 * Two boundary levels are used in the app:
 *   1. SchemaErrorBoundary — wraps DynamicClinicalForm (schema fetch failures)
 *   2. GlobalErrorBoundary — wraps the entire DiseaseProvider subtree
 */

import { Component } from 'react';
import SchemaErrorFallback from './SchemaErrorFallback';

/**
 * Generic Error Boundary with configurable fallback.
 *
 * Usage:
 *   <ErrorBoundary
 *     fallback={<MyCustomFallback onRetry={...} />}
 *     onError={(error, info) => console.error(error, info)}
 *   >
 *     <MyComponent />
 *   </ErrorBoundary>
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Log to console (or external service in production)
    console.error('[ErrorBoundary] Caught error:', error);
    if (info?.componentStack) {
      console.error('[ErrorBoundary] Component stack:', info.componentStack);
    }
    // Call optional onError prop
    if (this.props.onError) {
      this.props.onError(error, info);
    }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided, otherwise default
      if (this.props.fallback) {
        return typeof this.props.fallback === 'function'
          ? this.props.fallback({
              error: this.state.error,
              retry: this.handleRetry,
            })
          : this.props.fallback;
      }

      // Default fallback
      return (
        <SchemaErrorFallback
          error={this.state.error}
          onRetry={this.handleRetry}
          disease={this.props.disease}
        />
      );
    }

    return this.props.children;
  }
}

/**
 * SchemaErrorBoundary — pre-configured for schema-related errors.
 * Includes a disease prop for context in the fallback UI.
 */
export function SchemaErrorBoundary({ children, disease }) {
  return (
    <ErrorBoundary
      disease={disease}
      fallback={({ error, retry }) => (
        <SchemaErrorFallback
          error={error}
          onRetry={retry}
          disease={disease}
        />
      )}
    >
      {children}
    </ErrorBoundary>
  );
}

