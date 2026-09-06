"""
monitoring/otel_setup.py

OpenTelemetry setup for SubsurfaceIQ.
Instruments the RAG pipeline with spans, capturing tokens,
cost, and duration for each call.

Same pattern as Module 5 homework — but exports to Postgres
instead of SQLite, so Grafana can query directly.
"""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SpanExporter, SpanExportResult, SimpleSpanProcessor
)
from monitoring.db import get_connection


class PostgresSpanExporter(SpanExporter):
    """
    Custom OTel exporter: finished spans → Postgres spans table.

    WHY custom exporter vs auto-instrumentation:
    We need domain-specific attributes (query, well_id, search_mode)
    that auto-instrumentation doesn't capture. Manual set_attribute()
    in the RAG code gives us full control over what gets stored.
    """

    def export(self, spans):
        try:
            conn = get_connection()
            cur  = conn.cursor()

            for span in spans:
                attrs        = dict(span.attributes or {})
                duration_ms  = (span.end_time - span.start_time) / 1_000_000

                cur.execute("""
                    INSERT INTO spans
                    (span_name, query, well_id, search_mode,
                     input_tokens, output_tokens, cost,
                     duration_ms, start_time, end_time)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    span.name,
                    attrs.get("query"),
                    attrs.get("well_id"),
                    attrs.get("search_mode"),
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                    round(duration_ms, 3),
                    span.start_time,
                    span.end_time,
                ))

            conn.commit()
            cur.close()
            conn.close()
            return SpanExportResult.SUCCESS

        except Exception as e:
            print(f"OTel export error: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self): pass
    def force_flush(self, timeout_millis=None): return True


def setup_tracing() -> trace.Tracer:
    """
    Initialize OTel tracing. Call once at app startup.
    Returns a tracer — pass to VolveRAG or use directly.
    """
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(PostgresSpanExporter())
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("subsurface-iq")


# Module-level tracer — import and use anywhere
tracer = setup_tracing()