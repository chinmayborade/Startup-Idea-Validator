from pydantic import BaseModel, Field
from typing import List


class MarketAnalysis(BaseModel):
      market_size: str = Field(description="Estimated TAM/SAM/SOM in $ billions")
      growth_size:str = Field(description="Annual market growth rate percentage")
      key_tends:str= Field(description="3-5 market trends supporting this idea")
      market_maturity:str= Field(description="Emerging/Growing/Mature/Declining")
      score:int =Field(description="Market opportunity score 1=10")


class CompetitorAnalysis(BaseModel):
      direct_competitors:List[str]=Field(description="Direct competitors")
      indirect_competitors:List[str] =Field(description="Indirect or adjacent competitors")
      competitive_advantages:List[str]=Field(description="Potential advantages over competitors")
      competitive_moat:List[str]=Field(description="Suggestive defensible moat strategy")
      score:int =Field(description="Competitive landscape score 1-10")


class FeasibilityAnalysis(BaseModel):
      technical_complexity:str = Field(description="Low/Medium/high")
      estimated_mvp_timeline:str= Field(description="Estimated months to build mvp")
      key_technical_challenges:List[str] = Field(description="Main techincal hurdles")
      required_team_skills:List[str]= Field(description="Key skills needed in founding team")
      score:int = Field(description="Feasibility Score 1-10")

class RevenueAnalysis(BaseModel):
       recommended_model : str = Field(description="Best Revenue model")
       pricing_strategy:str = Field(description="Suggested pricing approach")
       unit_economics:str = Field(description="Unit economics to track LTV,CAC,etc")
       monetization_timeline:str = Field(description="How quickly revenue can be gerneated")
       score:int=Field(description="Revenue potential score 1-10 ")


class RiskAnalysis(BaseModel):
       critical_risks:List[str]=Field(description="5 critical risks that could affect the startup")
       regulatory_concerns:List[str] =Field(description="Regulatory or legal risks to watch")
       monetization_risks:List[str]=Field(description="5 monetory risks that could affect the startup")

       score:int = Field(description="Risk Score 1-10")

class ValidationSummary(BaseModel):
    overall_score: int = Field(description="Overall viability score 0-100")
    verdict: str = Field(description="Go / Pivot / Kill verdict with one sentence reason")
    elevator_pitch: str = Field(description="Refined 2-sentence elevator pitch")
    next_steps: List[str] = Field(description="Top 3 immediate action items for the founder")
    biggest_assumption: str = Field(description="The single biggest assumption to validate first")

