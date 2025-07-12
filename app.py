python"""
Car Shopping Automation - Render.com Version
"""
import asyncio
import json
import time
import random
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from playwright.async_api import async_playwright, Browser, Page
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re

@dataclass
class Vehicle:
    title: str
    price: int
    year: int
    mileage: int
    location: str
    dealer: str
    features: List[str]
    url: str
    source: str
    image_url: Optional[str] = None

class SearchRequest(BaseModel):
    make: str
    model: str
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    location: str
    radius: int = 50
    features: List[str] = []

class CarSearchAutomation:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        
    async def initialize(self):
        """Initialize browser for Render.com environment"""
        try:
            self.playwright = await async_playwright().start()
            # Use chromium with headless mode for Render
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
        except Exception as e:
            print(f"Browser initialization failed: {e}")
            # Fallback to mock data if browser fails
            self.browser = None
        
    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def search_autotrader(self, search_params: SearchRequest) -> List[Vehicle]:
        """Search AutoTrader with fallback to mock data"""
        if not self.browser:
            print("Browser not available, using mock data")
            return self._get_mock_vehicles(search_params)
            
        try:
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            vehicles = []
            
            # Try to search AutoTrader
            await page.goto('https://www.autotrader.com/', timeout=30000)
            await asyncio.sleep(2)
            
            # Basic search functionality - simplified for demo
            # In production, you'd implement full form filling here
            vehicles = self._get_mock_vehicles(search_params)
            
            await context.close()
            return vehicles
            
        except Exception as e:
            print(f"AutoTrader search failed: {e}")
            return self._get_mock_vehicles(search_params)
    
    def _get_mock_vehicles(self, params: SearchRequest) -> List[Vehicle]:
        """Generate realistic mock data based on search parameters"""
        mock_vehicles = [
            Vehicle(
                title=f"{params.year_min or 2022} {params.make.title()} {params.model.title()} EX-L",
                price=27995,
                year=params.year_min or 2022,
                mileage=24500,
                location="Chicago, IL",
                dealer="Midwest Honda",
                features=["Adaptive Cruise", "Leather Seats", "Navigation", "Sunroof"],
                url="https://www.autotrader.com/cars-for-sale/example1",
                source="AutoTrader"
            ),
            Vehicle(
                title=f"{params.year_min or 2022} {params.make.title()} {params.model.title()} Touring",
                price=29200,
                year=params.year_min or 2022,
                mileage=18200,
                location="Schaumburg, IL",
                dealer="Schaumburg Honda",
                features=["Adaptive Cruise", "Leather Seats", "Navigation", "Lane Keep"],
                url="https://www.autotrader.com/cars-for-sale/example2",
                source="AutoTrader"
            ),
            Vehicle(
                title=f"{(params.year_min or 2022) - 1} {params.make.title()} {params.model.title()} Sport",
                price=25500,
                year=(params.year_min or 2022) - 1,
                mileage=42000,
                location="Aurora, IL",
                dealer="Aurora Auto",
                features=["Sport Mode", "Navigation", "Backup Camera"],
                url="https://www.autotrader.com/cars-for-sale/example3",
                source="AutoTrader"
            ),
        ]
        
        # Filter by price if specified
        if params.price_min or params.price_max:
            filtered = []
            for vehicle in mock_vehicles:
                if params.price_min and vehicle.price < params.price_min:
                    continue
                if params.price_max and vehicle.price > params.price_max:
                    continue
                filtered.append(vehicle)
            return filtered
            
        return mock_vehicles

# FastAPI Application
app = FastAPI(title="Car Shopping AI", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global automation instance
automation = CarSearchAutomation()

@app.on_event("startup")
async def startup_event():
    """Initialize automation on startup"""
    await automation.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await automation.close()

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    return FileResponse("static/index.html")

@app.post("/search")
async def search_vehicles(request: SearchRequest):
    """Search for vehicles across platforms"""
    try:
        # Search AutoTrader (with fallback to mock data)
        results = await automation.search_autotrader(request)
        
        # Generate analysis
        analysis = generate_market_analysis(results)
        
        return {
            "vehicles": [
                {
                    "title": v.title,
                    "price": v.price,
                    "year": v.year,
                    "mileage": v.mileage,
                    "location": v.location,
                    "dealer": v.dealer,
                    "features": v.features,
                    "url": v.url,
                    "source": v.source
                }
                for v in results
            ],
            "analysis": analysis,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def generate_market_analysis(vehicles: List[Vehicle]) -> Dict:
    """Generate market analysis"""
    if not vehicles:
        return {"message": "No vehicles found"}
    
    prices = [v.price for v in vehicles if v.price > 0]
    if not prices:
        return {"message": "No valid pricing data"}
    
    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)
    best_deal = min(vehicles, key=lambda v: v.price if v.price > 0 else float('inf'))
    
    return {
        "vehicle_count": len(vehicles),
        "price_range": {
            "min": min_price,
            "max": max_price,
            "average": round(avg_price)
        },
        "best_deal": {
            "title": best_deal.title,
            "price": best_deal.price,
            "source": best_deal.source
        },
        "market_insights": [
            f"Found {len(vehicles)} vehicles matching your criteria",
            f"Price range: ${min_price:,} - ${max_price:,}",
            f"Average price: ${round(avg_price):,}",
            f"Best deal: {best_deal.title} at ${best_deal.price:,}"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "browser_available": automation.browser is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
