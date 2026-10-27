"""
Clinical Results Service (RF-07, RF-08, RF-09, RF-10, RF-11)
==============================================================
Manages prediction results, clinical review, history, and report generation.
"""

from datetime import datetime, timezone
import zoneinfo
from typing import Dict, List, Optional

from domain.interfaces.contracts import (
    IResultRepository,
    IImageRepository,
    IAuditRepository,
    PredictionResult,
    PredictionStatus,
    AuditEntry,
    AuditAction,
)
from config.settings import REPORTS_DIR
from utils.logger import get_logger

logger = get_logger("ClinicalResultsService")

MEXICO_TZ = zoneinfo.ZoneInfo("America/Mexico_City")


def format_mexico_datetime(dt: Optional[datetime], is_es: bool = True) -> str:
    """
    Convierte una fecha en UTC o naive al huso horario de México (America/Mexico_City).
    Formato en 12 horas con indicador de zona CST:
      - Español: 26/10/2026 02:14 p. m. (CST)
      - Inglés:  2026-10-26 02:14 PM (CST)
    """
    if not dt:
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    m_dt = dt.astimezone(MEXICO_TZ)
    if is_es:
        am_pm = "p. m." if m_dt.hour >= 12 else "a. m."
        return f"{m_dt.strftime('%d/%m/%Y %I:%M')} {am_pm} (CST)"
    else:
        return m_dt.strftime("%Y-%m-%d %I:%M %p (CST)")


class ClinicalResultsService:
    """
    Manages the lifecycle of clinical prediction results.

    Responsibilities:
      - Persist classification results (RF-08).
      - Provide result visualization data (RF-07).
      - Support clinical review (confirm/correct) by specialists (RF-09).
      - Generate structured clinical reports in PDF (RF-10).
      - Maintain searchable prediction history (RF-11).
    """

    def __init__(
        self,
        result_repository: IResultRepository,
        image_repository: IImageRepository,
        audit_repository: IAuditRepository,
    ):
        self._result_repo = result_repository
        self._image_repo = image_repository
        self._audit_repo = audit_repository

    def save_prediction(
        self,
        image_id: int,
        classification: str,
        confidence_score: float,
        model_name: str,
        model_version: str,
        processing_time_ms: float,
        affected_vertebrae: Optional[str] = None,
        curve_type: Optional[str] = None,
        curve_detail: Optional[str] = None,
    ) -> PredictionResult:
        """Save a new prediction result to the database (RF-08)."""
        result = PredictionResult(
            image_id=image_id,
            classification=classification,
            confidence_score=confidence_score,
            model_name=model_name,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
            status=PredictionStatus.PENDING_REVIEW,
            affected_vertebrae=affected_vertebrae,
            curve_type=curve_type,
            curve_detail=curve_detail,
        )

        saved = self._result_repo.save(result)
        logger.info(f"Prediction saved: ID={saved.id} for image {image_id} (Vertebrae: {affected_vertebrae})")
        return saved

    def review_prediction(
        self,
        result_id: int,
        reviewer_id: int,
        status: PredictionStatus,
        notes: str,
    ) -> Optional[PredictionResult]:
        """
        Submit a clinical review for a prediction (RF-09).

        A radiologist or specialist confirms or corrects the AI prediction,
        and the review is recorded with full traceability.
        """
        updated = self._result_repo.update_review(
            result_id=result_id,
            status=status,
            reviewer_id=reviewer_id,
            notes=notes,
        )

        if updated:
            # Audit trail: prediction reviewed
            self._audit_repo.append(AuditEntry(
                action=AuditAction.PREDICTION_REVIEWED,
                user_id=reviewer_id,
                resource_type="prediction_result",
                resource_id=result_id,
                details=(
                    f"Prediction {result_id} reviewed: {status.value} | "
                    f"Notes: {notes}"
                ),
            ))
            logger.info(f"Prediction {result_id} reviewed: {status.value}")

        return updated

    def get_result(self, result_id: int) -> Optional[PredictionResult]:
        """Retrieve a specific prediction result (RF-07)."""
        return self._result_repo.find_by_id(result_id)

    def get_result_by_image(self, image_id: int) -> Optional[PredictionResult]:
        """Retrieve the latest prediction for a given image."""
        return self._result_repo.find_by_image_id(image_id)

    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[PredictionStatus] = None,
    ) -> List[PredictionResult]:
        """Retrieve prediction history with optional filtering (RF-11)."""
        return self._result_repo.find_all(limit=limit, offset=offset, status=status)

    def generate_report(
        self,
        result_id: int,
        reviewer_name: Optional[str] = None,
        language: str = "en",
        user_id: Optional[int] = None,
    ) -> str:
        """
        Generate a structured clinical report in PDF format (RF-10).

        Supports full English ("en") or full Spanish ("es") reports based on
        the system language selection. Ensures zero text clipping or cursor
        misplacement (fixing any trailing text glitch).
        """
        result = self._result_repo.find_by_id(result_id)
        if not result:
            raise ValueError(f"Prediction result {result_id} not found")

        image = self._image_repo.find_by_id(result.image_id)
        if not image:
            raise ValueError(f"Image {result.image_id} not found")

        is_es = str(language).lower().strip() == "es"

        try:
            from fpdf import FPDF
            import re

            # Helper functions for localization
            def get_localized_classification() -> str:
                cls_raw = result.classification or ""
                upper_cls = cls_raw.upper()
                if "NO ESCOLIOSIS" in upper_cls or "NO SCOLIOSIS" in upper_cls or "NORMAL" in upper_cls:
                    return "NORMAL (SIN ESCOLIOSIS)" if is_es else "NORMAL (NO SCOLIOSIS)"
                elif "EN C" in upper_cls or "C-CURVE" in upper_cls:
                    return "ESCOLIOSIS GRAVE (CURVA EN C)" if is_es else "SEVERE SCOLIOSIS (C-SHAPED CURVE)"
                elif "EN S" in upper_cls or "S-CURVE" in upper_cls:
                    return "ESCOLIOSIS SEVERA (CURVA EN S)" if is_es else "SEVERE SCOLIOSIS (S-SHAPED CURVE)"
                return cls_raw

            def get_localized_vertebrae() -> str:
                v = result.affected_vertebrae or ""
                if not v or "ninguna" in v.lower() or "none" in v.lower():
                    return "Ninguna (Alineación anatómica normal)" if is_es else "None (Normal anatomical alignment)"
                return v

            def get_localized_curve_type() -> str:
                ct = result.curve_type or ""
                if "en C" in ct or "C-curve" in ct or "C-Shaped" in ct:
                    return "Curvatura en C (Escoliosis Simple)" if is_es else "C-Shaped Curve (Simple Scoliosis)"
                elif "en S" in ct or "S-curve" in ct or "S-Shaped" in ct:
                    return "Curvatura en S (Escoliosis Doble / Estructural)" if is_es else "S-Shaped Curve (Double / Structural Scoliosis)"
                elif "normal" in ct.lower() or "fisiol" in ct.lower():
                    return "Alineación Fisiológica Normal" if is_es else "Normal Physiological Alignment"
                return ct if is_es else "Unspecified"

            def get_localized_curve_detail() -> str:
                cd = result.curve_detail or ""
                if is_es:
                    if not cd:
                        return "Sin observaciones anatómicas adicionales registradas."
                    cd_es = cd.replace("Simple C-shaped curvature in thoracic segment with apex at", "Curvatura simple en forma de 'C' en segmento torácico con ápex en")
                    cd_es = cd_es.replace("Double S-shaped curvature. Primary apex at", "Doble curvatura en forma de 'S'. Ápice principal en")
                    cd_es = cd_es.replace("Normal physiological vertebral alignment without radiological signs of scoliosis.", "Alineación vertebral fisiológica normal sin signos radiológicos de escoliosis.")
                    return cd_es
                else:
                    if not cd:
                        return "No additional anatomical observations recorded."
                    if "simple en forma" in cd.lower() or "en c" in cd.lower():
                        m = re.search(r"[áa]pex\s+en\s+([A-Za-z0-9]+)", cd, re.IGNORECASE)
                        apex = m.group(1) if m else "T9"
                        return f"Simple C-shaped curvature in thoracic segment with apex at {apex}."
                    elif "doble" in cd.lower() or "en s" in cd.lower():
                        m = re.search(r"([áa]pice|[áa]pex).*?([TLC]\d+)", cd, re.IGNORECASE)
                        apex = m.group(2) if m else "T11"
                        return f"Double S-shaped curvature. Primary curve apex at {apex}."
                    elif "fisiol" in cd.lower() or "normal" in cd.lower():
                        return "Normal physiological vertebral alignment without radiological signs of scoliosis."
                    return cd

            def get_localized_status() -> str:
                st = result.status.value if hasattr(result.status, "value") else str(result.status)
                if st == "pending_review":
                    return "Pendiente de Revisión Médica" if is_es else "Pending Specialist Review"
                elif st == "confirmed":
                    return "Confirmado por Especialista" if is_es else "Confirmed by Specialist"
                elif st == "corrected":
                    return "Corregido por Especialista" if is_es else "Corrected by Specialist"
                return st

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_margins(15, 15, 15)

            # ── Top Header Banner (Medical Navy #1E3A8A) ──
            pdf.set_fill_color(30, 58, 138)
            pdf.rect(15, 15, 180, 24, "F")

            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_xy(15, 19)
            doc_title = "REPORTE DE CLASIFICACIÓN CLÍNICA" if is_es else "CLINICAL CLASSIFICATION REPORT"
            pdf.cell(180, 8, doc_title, align="C", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 9)
            subtitle_text = (
                "Sistema de Diagnóstico de Escoliosis Asistido por Inteligencia Artificial"
                if is_es else
                "AI-Assisted Spinal Radiograph Scoliosis Diagnostic System"
            )
            pdf.cell(180, 5, subtitle_text, align="C", new_x="LMARGIN", new_y="NEXT")

            pdf.set_y(44)

            # ── Metadata Bar ──
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(107, 114, 128)
            label_id = "ID del Reporte:" if is_es else "Report ID:"
            label_date = "Fecha de Emisión:" if is_es else "Issue Date:"

            now_str = format_mexico_datetime(datetime.now(timezone.utc), is_es=is_es)

            pdf.cell(30, 6, label_id)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(31, 41, 55)
            pdf.cell(50, 6, f"RPT-{result_id:04d}")

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(107, 114, 128)
            pdf.cell(35, 6, label_date)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(31, 41, 55)
            pdf.cell(65, 6, now_str, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(3)

            # ── Section Helpers ──
            def section_header(title: str):
                pdf.set_fill_color(240, 244, 248)
                pdf.set_text_color(30, 58, 138)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_x(15)
                pdf.cell(180, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

            def key_value(key: str, val: str, is_highlight: bool = False):
                pdf.set_x(17)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(75, 85, 99)
                pdf.cell(55, 6, key)

                pdf.set_font("Helvetica", "B" if is_highlight else "", 9)
                if is_highlight:
                    pdf.set_text_color(30, 58, 138)
                else:
                    pdf.set_text_color(17, 24, 39)
                pdf.cell(123, 6, val, new_x="LMARGIN", new_y="NEXT")
                pdf.set_x(15)

            def key_multiline(key: str, val: str):
                pdf.set_x(17)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(75, 85, 99)
                pdf.cell(55, 6, key)

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(17, 24, 39)
                # Ensure cursor returns safely to left margin after multiline text
                pdf.multi_cell(123, 6, val, new_x="LMARGIN", new_y="NEXT")
                pdf.set_x(15)

            # ── Section 1: Radiograph Image Information ──
            sec1_title = "1. INFORMACIÓN DE LA RADIOGRAFÍA" if is_es else "1. RADIOGRAPH IMAGE INFORMATION"
            section_header(sec1_title)

            key_value("Nombre de Archivo:" if is_es else "Filename:", image.filename)
            key_value("Formato:" if is_es else "Format:", str(image.format or ".jpg").upper())
            key_value("Dimensiones:" if is_es else "Dimensions:", f"{image.width} x {image.height} px")
            up_str = format_mexico_datetime(image.uploaded_at, is_es=is_es)
            key_value("Fecha de Carga:" if is_es else "Upload Date:", up_str)
            pdf.ln(3)

            # ── Section 2: AI Diagnostic Evaluation ──
            sec2_title = "2. EVALUACIÓN DIAGNÓSTICA POR INTELIGENCIA ARTIFICIAL" if is_es else "2. SCOLIOSIS AI DIAGNOSTIC EVALUATION"
            section_header(sec2_title)

            key_value("Diagnóstico / Clasificación:" if is_es else "Diagnosis / Classification:", get_localized_classification(), is_highlight=True)
            key_value("Vértebras Afectadas:" if is_es else "Affected Vertebrae:", get_localized_vertebrae())
            key_value("Tipo de Curva:" if is_es else "Curve Type:", get_localized_curve_type())
            key_multiline("Detalle Clínico:" if is_es else "Clinical Detail:", get_localized_curve_detail())

            # Fixed cursor ensures confidence level renders cleanly without truncation
            conf_str = f"{result.confidence_score:.2%}"
            key_value("Nivel de Confianza:" if is_es else "Confidence Score:", conf_str)

            model_str = f"{result.model_name} v{result.model_version}"
            key_value("Modelo de IA:" if is_es else "AI Model Architecture:", model_str)

            time_str = f"{result.processing_time_ms:.1f} ms"
            key_value("Tiempo de Procesamiento:" if is_es else "Processing Latency:", time_str)

            pred_str = format_mexico_datetime(result.predicted_at, is_es=is_es)
            key_value("Fecha de Análisis:" if is_es else "Analysis Timestamp:", pred_str)
            pdf.ln(3)

            # ── Section 3: Clinical Review & Specialist Validation ──
            sec3_title = "3. REVISIÓN Y VALIDACIÓN CLÍNICA" if is_es else "3. CLINICAL REVIEW & SPECIALIST VALIDATION"
            section_header(sec3_title)

            key_value("Estado de Revisión:" if is_es else "Review Status:", get_localized_status())

            default_rev = "Pendiente de Asignación Médica" if is_es else "Pending Specialist Assignment"
            actual_rev = reviewer_name.strip() if reviewer_name and reviewer_name.strip() else default_rev
            key_value("Especialista Revisor:" if is_es else "Specialist Reviewer:", actual_rev)

            notes = result.reviewer_notes
            if not notes or not notes.strip() or notes == "No notes provided" or notes == "Sin notas adicionales":
                notes_str = "Sin observaciones clínicas adicionales registradas." if is_es else "No additional clinical observations provided."
            else:
                notes_str = notes.strip()
            key_multiline("Notas Clínicas:" if is_es else "Clinical Notes:", notes_str)
            pdf.ln(5)

            # ── Section 4: Legal Disclaimer ──
            pdf.set_fill_color(250, 250, 250)
            pdf.set_draw_color(220, 225, 230)
            pdf.rect(15, pdf.get_y(), 180, 23, "DF")

            pdf.set_xy(18, pdf.get_y() + 2)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(107, 114, 128)
            disc_header = "AVISO LEGAL Y DESCARGO DE RESPONSABILIDAD MÉDICA:" if is_es else "CLINICAL DECISION SUPPORT DISCLAIMER:"
            pdf.cell(174, 4, disc_header, new_x="LMARGIN", new_y="NEXT")

            pdf.set_x(18)
            pdf.set_font("Helvetica", "I", 7)
            disc_text = (
                "Este reporte es emitido por un sistema de apoyo al diagnóstico médico asistido por inteligencia artificial. "
                "Los resultados constituyen una segunda opinión técnica orientativa y NO sustituyen bajo ninguna circunstancia "
                "el juicio clínico, diagnóstico o prescripción de un especialista médico colegiado. Todos los hallazgos deben ser "
                "correlacionados integralmente con el historial del paciente."
                if is_es else
                "This report is generated by an AI-assisted clinical decision support system. The AI classification "
                "serves as a secondary technical opinion and does NOT replace the professional medical judgment, diagnosis, "
                "or treatment plan of a licensed medical specialist or radiologist. All findings must be clinically correlated "
                "with patient history and physical examination."
            )
            pdf.multi_cell(174, 3.5, disc_text, new_x="LMARGIN", new_y="NEXT")

            # Save report file
            now_mx = datetime.now(timezone.utc).astimezone(MEXICO_TZ)
            report_filename = f"report_{result_id:04d}_{language}_{now_mx.strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = str(REPORTS_DIR / report_filename)
            pdf.output(report_path)

            # Audit trail
            self._audit_repo.append(AuditEntry(
                action=AuditAction.REPORT_GENERATED,
                user_id=user_id,
                resource_type="clinical_report",
                resource_id=result_id,
                details=f"Report generated: {report_filename} ({language.upper()})",
            ))

            logger.info(f"Clinical report generated: {report_path} [Lang: {language.upper()}]")
            return report_path

        except ImportError:
            logger.warning("fpdf2 not installed — generating text report instead")
            return self._generate_text_report(result, image, reviewer_name, language)

    def _generate_text_report(
        self,
        result: PredictionResult,
        image,
        reviewer_name: Optional[str],
        language: str = "en",
    ) -> str:
        """Fallback: generate a text-based report if PDF library is unavailable."""
        is_es = str(language).lower().strip() == "es"
        now_str = format_mexico_datetime(datetime.now(timezone.utc), is_es=is_es)
        up_str = format_mexico_datetime(image.uploaded_at, is_es=is_es)
        pred_str = format_mexico_datetime(result.predicted_at, is_es=is_es)

        report_filename = f"report_{result.id:04d}_{language}.txt"
        report_path = str(REPORTS_DIR / report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            if is_es:
                f.write("=" * 60 + "\n")
                f.write("    REPORTE CLÍNICO DE EVALUACIÓN DE ESCOLIOSIS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"ID del Reporte: RPT-{result.id:04d}\n")
                f.write(f"Fecha de Emisión: {now_str}\n")
                f.write(f"Archivo: {image.filename}\n")
                f.write(f"Fecha de Carga: {up_str}\n")
                f.write(f"Diagnóstico: {result.classification}\n")
                f.write(f"Vértebras Afectadas: {result.affected_vertebrae or 'Ninguna'}\n")
                f.write(f"Tipo de Curva: {result.curve_type or 'N/A'}\n")
                f.write(f"Detalle Clínico: {result.curve_detail or 'N/A'}\n")
                f.write(f"Nivel de Confianza: {result.confidence_score:.2%}\n")
                f.write(f"Modelo: {result.model_name} v{result.model_version}\n")
                f.write(f"Tiempo de Procesamiento: {result.processing_time_ms:.1f} ms\n")
                f.write(f"Fecha de Análisis: {pred_str}\n")
                f.write(f"Estado de Revisión: {result.status.value}\n")
                f.write(f"Especialista Revisor: {reviewer_name or 'Pendiente'}\n")
                f.write(f"Notas Clínicas: {result.reviewer_notes or 'Sin notas adicionales'}\n")
            else:
                f.write("=" * 60 + "\n")
                f.write("       CLINICAL SCOLIOSIS EVALUATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report ID: RPT-{result.id:04d}\n")
                f.write(f"Issue Date: {now_str}\n")
                f.write(f"Filename: {image.filename}\n")
                f.write(f"Upload Date: {up_str}\n")
                f.write(f"Diagnosis: {result.classification}\n")
                f.write(f"Affected Vertebrae: {result.affected_vertebrae or 'None'}\n")
                f.write(f"Curve Type: {result.curve_type or 'N/A'}\n")
                f.write(f"Clinical Detail: {result.curve_detail or 'N/A'}\n")
                f.write(f"Confidence Score: {result.confidence_score:.2%}\n")
                f.write(f"AI Model: {result.model_name} v{result.model_version}\n")
                f.write(f"Processing Time: {result.processing_time_ms:.1f} ms\n")
                f.write(f"Analysis Timestamp: {pred_str}\n")
                f.write(f"Review Status: {result.status.value}\n")
                f.write(f"Specialist Reviewer: {reviewer_name or 'Pending'}\n")
                f.write(f"Clinical Notes: {result.reviewer_notes or 'No notes provided'}\n")

        return report_path
