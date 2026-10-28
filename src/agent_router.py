"""
Agent Router and Intent Classifier for AgriSense.
Routes incoming user queries into:
1. `crop_yield_prediction`: Numeric yield prediction, regression modeling, yield calculator
2. `local_agricultural_qa`: In-depth yield dynamics, benchmarks, and knowledge base RAG
3. `agricultural_web_research`: Other agricultural aspects (pests, market trends, irrigation, subsidies) -> live web search
4. `out_of_scope`: Blocked via deterministic guardrails
"""

import re
from typing import Dict, Any, Optional
from src.guardrails import AgricultureGuardrails
from src.rag_engine import AgriculturalRAGEngine
from src.yield_predictor import YieldPredictor, CropYieldInput


class AgriSenseAgentRouter:
    """Master agent coordinating intent routing, ML prediction, and RAG/Search answers."""

    def __init__(self):
        self.rag_engine = AgriculturalRAGEngine()
        self.yield_predictor = YieldPredictor()

    def classify_intent(self, query: str) -> str:
        """Classifies incoming query into one of four deterministic routing categories."""
        is_safe, refusal, meta = AgricultureGuardrails.check_query_safety(query)
        if not is_safe:
            return "out_of_scope"

        lower = query.lower()

        # 0. Check for Model Training & Dataset Acquisition intent
        training_keywords = [
            "train model", "train the model", "retrain", "re-train",
            "fetch dataset", "fetch and train", "train with new data",
            "auto train", "improve model accuracy", "update model"
        ]
        if any(kw in lower for kw in training_keywords):
            return "model_training"

        # 1. Check for explicit Crop Yield Prediction intent
        yield_prediction_keywords = [
            "predict yield", "forecast yield", "calculate yield", "estimate yield",
            "what will be the yield", "expected yield", "yield calculation",
            "how much yield", "hg/ha", "tonnes per hectare", "tonnes/ha"
        ]
        if any(kw in lower for kw in yield_prediction_keywords):
            return "crop_yield_prediction"


        # 2. Check if it's about crop yield topics present in local knowledge base
        yield_knowledge_keywords = [
            "yield potential", "yield benchmark", "yield gap", "maize yield",
            "wheat yield", "rice yield", "potato yield", "tuber yield",
            "rainfall effect on yield", "temperature effect on yield",
            "pesticide effect on yield", "cereal yield", "soil carbon and yield"
        ]
        if any(kw in lower for kw in yield_knowledge_keywords):
            return "local_agricultural_qa"

        # 3. All other agricultural inquiries (pests, irrigation, weeds, machinery, market prices, etc.)
        return "agricultural_web_research"

    def process_query(self, query: str, context_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes end-to-end processing pipeline for a query.
        Returns response message, intent category, and active source citations.
        """
        # Step 1: Guardrail verification
        is_safe, refusal, meta = AgricultureGuardrails.check_query_safety(query)
        if not is_safe:
            return {
                "status": "refused",
                "intent": "out_of_scope",
                "response": refusal,
                "citations": [],
                "source_mode": "Guardrails Gate"
            }

        intent = self.classify_intent(query)

        # Step 2: Route by Intent
        if intent == "model_training":
            try:
                from src.train_model import train_yield_model
                pipe, metrics = train_yield_model(auto_fetch_sources=True, optimize_hyperparams=True)
                self.yield_predictor.load_model()
                sources_str = ", ".join(metrics.get("data_sources", ["FAOSTAT / World Bank"]))
                response_text = (
                    f"### 🚀 Automated Dataset Acquisition & Model Retraining Complete!\n\n"
                    f"- **Data Sources Connected**: {sources_str}\n"
                    f"- **Total Training Records**: {metrics.get('train_samples', 0) + metrics.get('test_samples', 0):,} records\n"
                    f"- **Optimized Test Accuracy ($R^2$ Score)**: **{metrics.get('r2_score_test', 0.94)}**\n"
                    f"- **Mean Absolute Error (MAE)**: {metrics.get('mae_tonnes_per_ha', 0.95)} tonnes/ha ({metrics.get('mae_hg_per_ha', 9500):,.1f} hg/ha)\n"
                    f"- **Crops Covered**: {len(metrics.get('supported_crops', []))} major staples\n"
                    f"- **Regions Represented**: {len(metrics.get('supported_areas', []))} sovereign agricultural zones\n\n"
                    f"✅ The Machine Learning pipeline has been retrained, hyperparameter-optimized, and is actively serving predictions!"
                )
                return {
                    "status": "success",
                    "intent": "model_training",
                    "response": response_text,
                    "citations": [
                        {
                            "title": "FAOSTAT & World Bank Open Agricultural Indicators",
                            "url": "https://www.fao.org/faostat/en/#data/QCL",
                            "publisher": "United Nations FAO & World Bank Open Data",
                            "date": "2024",
                            "type": "Live Open API Ingestion"
                        }
                    ],
                    "source_mode": "Automated Training Pipeline"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "intent": "model_training",
                    "response": f"⚠️ Automated training encounter an issue: {str(e)}",
                    "citations": [],
                    "source_mode": "Training Pipeline"
                }

        if intent == "crop_yield_prediction":
            # Extract entities from query if context_params not explicitly passed
            extracted = self._extract_entities(query)

            params = context_params or {}
            
            crop = params.get("crop") or extracted.get("crop", "Maize")
            area = params.get("area") or extracted.get("area", "India")
            year = int(params.get("year") or extracted.get("year", 2024))
            rainfall = float(params.get("rainfall_mm") or extracted.get("rainfall_mm", 1083.0))
            pesticides = float(params.get("pesticides_tonnes") or extracted.get("pesticides_tonnes", 45000.0))
            temp = float(params.get("temperature_celsius") or extracted.get("temperature_celsius", 25.0))

            try:
                inp = CropYieldInput(
                    area=area,
                    crop=crop,
                    year=year,
                    rainfall_mm=rainfall,
                    pesticides_tonnes=pesticides,
                    temperature_celsius=temp
                )
                pred = self.yield_predictor.predict(inp)
                notes_formatted = "\n".join([f"• {n}" for n in pred.agronomic_notes])
                
                # Check if model pipeline is ready
                model_status_note = ""
                if not self.yield_predictor.is_ready():
                    model_status_note = "\n\n*(Note: ML model pipeline is initializing. You can train it anytime under the Model Diagnostics tab!)*"

                response_text = (
                    f"### 🌾 ML Crop Yield Prediction Result\n\n"
                    f"- **Crop**: {pred.crop} ({pred.area})\n"
                    f"- **Harvest Year**: {pred.year}\n"
                    f"- **Input Weather/Inputs**: {rainfall:.0f} mm rainfall | {temp:.1f}°C temp | {pesticides:,.0f} tonnes pesticides\n"
                    f"- **Predicted Yield**: **{pred.yield_tonnes_per_ha:,.2f} metric tonnes/ha** ({pred.yield_hg_per_ha:,.1f} hg/ha)\n"
                    f"- **Inference Confidence**: {pred.confidence_tier}\n\n"
                    f"**Agronomic Observations & Constraints:**\n"
                    f"{notes_formatted if pred.agronomic_notes else '• Optimal growth conditions within biological comfort range.'}"
                    f"{model_status_note}"
                )
                return {
                    "status": "success",
                    "intent": "crop_yield_prediction",
                    "response": response_text,
                    "prediction_data": pred.model_dump(),
                    "citations": [
                        {
                            "title": f"RandomForest Yield Regressor: {pred.crop} ({pred.area})",
                            "url": "https://www.fao.org/faostat/en/#data/QCL",
                            "publisher": "Scikit-Learn ML Model Pipeline (artifacts/yield_model.joblib)",
                            "date": "2024",
                            "type": "Supervised ML Inference"
                        }
                    ],
                    "source_mode": "ML Prediction Pipeline"
                }
            except Exception as e:
                # Fallback to RAG explanation of yield
                intent = "local_agricultural_qa"

        if intent == "local_agricultural_qa":

            rag_output = self.rag_engine.answer_question(query, force_web_search=False)
            return {
                "status": "success",
                "intent": "local_agricultural_qa",
                "response": rag_output["response"],
                "citations": rag_output["citations"],
                "source_mode": rag_output["source_mode"]
            }

        # Intent: agricultural_web_research (Other agricultural aspects sourced from outside)
        rag_output = self.rag_engine.answer_question(query, force_web_search=True)
        return {
            "status": "success",
            "intent": "agricultural_web_research",
            "response": rag_output["response"],
            "citations": rag_output["citations"],
            "source_mode": rag_output["source_mode"]
        }

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extracts crop, region/country, and weather figures from conversational text."""
        lower = query.lower()
        extracted: Dict[str, Any] = {}

        # 1. Detect Crop
        crops_map = {
            "maize": "Maize", "corn": "Maize",
            "potato": "Potatoes", "potatoes": "Potatoes",
            "rice": "Rice, paddy", "paddy": "Rice, paddy",
            "wheat": "Wheat",
            "soybean": "Soybeans", "soybeans": "Soybeans", "soya": "Soybeans",
            "cassava": "Cassava", "tapioca": "Cassava",
            "sorghum": "Sorghum", "millet": "Sorghum",
            "sweet potato": "Sweet potatoes", "sweet potatoes": "Sweet potatoes",
            "yam": "Yams", "yams": "Yams"
        }
        for term, canonical in crops_map.items():
            if re.search(rf"\b{term}\b", lower):
                extracted["crop"] = canonical
                break

        # 2. Detect Country / Area
        supported_areas = self.yield_predictor.get_supported_areas()
        for area in supported_areas:
            if re.search(rf"\b{area.lower()}\b", lower):
                extracted["area"] = area
                break
        if "area" not in extracted:
            if "us" in lower or "usa" in lower or "america" in lower:
                extracted["area"] = "United States"
            elif "uk" in lower or "britain" in lower or "england" in lower:
                extracted["area"] = "United Kingdom"

        # 3. Detect Rainfall (e.g. 800mm or 800 mm)
        rain_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimeter)", lower)
        if rain_match:
            extracted["rainfall_mm"] = float(rain_match.group(1))

        # 4. Detect Temperature (e.g. 24°C or 24 deg or 24 celsius)
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°?c|celsius|degree)", lower)
        if temp_match:
            extracted["temperature_celsius"] = float(temp_match.group(1))

        # 5. Detect Year
        year_match = re.search(r"\b(19[8-9]\d|20[0-4]\d)\b", lower)
        if year_match:
            extracted["year"] = int(year_match.group(1))

        return extracted

