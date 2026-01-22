"""
Sentiment Analysis Module for Stock Prediction
==============================================

This module analyzes news articles, social media, and analyst reports
to extract sentiment signals that can improve stock price predictions.

Components:
- News sentiment extraction (headlines, article content)
- Social media sentiment (when available)
- Analyst recommendation aggregation
- Real-time news integration via web search

Author: Talal Alkhaled
License: MIT
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests


@dataclass
class NewsArticle:
    """Container for a news article."""
    source: str
    headline: str
    description: Optional[str]
    url: Optional[str]
    published_at: datetime
    sentiment: str  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    sentiment_score: float  # -1.0 to 1.0
    relevance: float  # 0.0 to 1.0


@dataclass
class SentimentResult:
    """Aggregated sentiment analysis result."""
    overall_sentiment: str
    sentiment_score: float
    confidence: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    article_count: int
    key_themes: List[str]
    reasoning: str


class SentimentAnalyzer:
    """
    Analyzes sentiment from multiple sources for stock prediction.
    
    Uses a combination of:
    - Rule-based sentiment scoring (keywords)
    - News API integration (Marketaux, Finnhub News)
    - AI-powered analysis (via external API)
    
    The sentiment score is combined with technical indicators
    to improve prediction accuracy.
    """
    
    def __init__(
        self,
        marketaux_key: Optional[str] = None,
        finnhub_key: Optional[str] = None
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            marketaux_key: Marketaux API key for news
            finnhub_key: Finnhub API key for company news
        """
        self.marketaux_key = marketaux_key
        self.finnhub_key = finnhub_key
        
        # Sentiment keywords (simplified - production uses more sophisticated NLP)
        self.positive_keywords = [
            'surge', 'soar', 'rally', 'gain', 'jump', 'rise', 'climb',
            'bullish', 'upgrade', 'beat', 'exceed', 'strong', 'growth',
            'profit', 'revenue', 'outperform', 'buy', 'opportunity',
            'breakthrough', 'innovation', 'positive', 'success', 'record',
            'momentum', 'optimistic', 'upside', 'expansion', 'partnership'
        ]
        
        self.negative_keywords = [
            'plunge', 'crash', 'drop', 'fall', 'decline', 'sink', 'tumble',
            'bearish', 'downgrade', 'miss', 'weak', 'loss', 'concern',
            'risk', 'warning', 'sell', 'underperform', 'cut', 'layoff',
            'lawsuit', 'investigation', 'negative', 'failure', 'slowdown',
            'pessimistic', 'downside', 'uncertainty', 'volatility', 'crisis'
        ]
        
        # Amplifier words
        self.amplifiers = ['very', 'extremely', 'significantly', 'sharply', 'dramatically']
        self.diminishers = ['slightly', 'marginally', 'somewhat', 'modestly']
    
    def fetch_news(
        self,
        symbol: str,
        limit: int = 10
    ) -> List[NewsArticle]:
        """
        Fetch recent news articles for a stock.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles to fetch
            
        Returns:
            List of NewsArticle objects
        """
        articles = []
        
        # Try Marketaux first
        if self.marketaux_key:
            try:
                marketaux_articles = self._fetch_marketaux(symbol, limit)
                articles.extend(marketaux_articles)
            except Exception as e:
                print(f"Marketaux fetch failed: {e}")
        
        # Try Finnhub news
        if self.finnhub_key and len(articles) < limit:
            try:
                finnhub_articles = self._fetch_finnhub_news(symbol, limit - len(articles))
                articles.extend(finnhub_articles)
            except Exception as e:
                print(f"Finnhub news fetch failed: {e}")
        
        return articles[:limit]
    
    def _fetch_marketaux(self, symbol: str, limit: int) -> List[NewsArticle]:
        """Fetch news from Marketaux API."""
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'symbols': symbol,
            'api_token': self.marketaux_key,
            'limit': limit,
            'language': 'en'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data.get('data', []):
            # Analyze sentiment for each article
            headline = item.get('title', '')
            description = item.get('description', '')
            sentiment, score = self.analyze_text(f"{headline} {description}")
            
            article = NewsArticle(
                source=item.get('source', 'Unknown'),
                headline=headline,
                description=description,
                url=item.get('url'),
                published_at=datetime.fromisoformat(
                    item.get('published_at', datetime.now().isoformat()).replace('Z', '+00:00')
                ),
                sentiment=sentiment,
                sentiment_score=score,
                relevance=1.0  # Marketaux pre-filters by relevance
            )
            articles.append(article)
        
        return articles
    
    def _fetch_finnhub_news(self, symbol: str, limit: int) -> List[NewsArticle]:
        """Fetch news from Finnhub API."""
        url = f"https://finnhub.io/api/v1/company-news"
        
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        params = {
            'symbol': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data[:limit]:
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            sentiment, score = self.analyze_text(f"{headline} {summary}")
            
            article = NewsArticle(
                source=item.get('source', 'Unknown'),
                headline=headline,
                description=summary,
                url=item.get('url'),
                published_at=datetime.fromtimestamp(item.get('datetime', 0)),
                sentiment=sentiment,
                sentiment_score=score,
                relevance=0.8
            )
            articles.append(article)
        
        return articles
    
    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a text using keyword-based approach.
        
        This is a simplified version. Production uses more sophisticated
        NLP models and AI-powered analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment label, sentiment score)
        """
        if not text:
            return 'NEUTRAL', 0.0
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = 0
        negative_count = 0
        amplifier_boost = 0
        
        for i, word in enumerate(words):
            # Check for positive keywords
            if word in self.positive_keywords:
                boost = 1.0
                # Check for amplifiers before this word
                if i > 0 and words[i-1] in self.amplifiers:
                    boost = 1.5
                elif i > 0 and words[i-1] in self.diminishers:
                    boost = 0.5
                positive_count += boost
            
            # Check for negative keywords
            if word in self.negative_keywords:
                boost = 1.0
                if i > 0 and words[i-1] in self.amplifiers:
                    boost = 1.5
                elif i > 0 and words[i-1] in self.diminishers:
                    boost = 0.5
                negative_count += boost
        
        # Calculate score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            return 'NEUTRAL', 0.0
        
        score = (positive_count - negative_count) / total
        
        # Determine sentiment label
        if score > 0.2:
            sentiment = 'POSITIVE'
        elif score < -0.2:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        return sentiment, score
    
    def aggregate_sentiment(
        self,
        articles: List[NewsArticle]
    ) -> SentimentResult:
        """
        Aggregate sentiment from multiple articles.
        
        Args:
            articles: List of NewsArticle objects
            
        Returns:
            Aggregated SentimentResult
        """
        if not articles:
            return SentimentResult(
                overall_sentiment='NEUTRAL',
                sentiment_score=0.0,
                confidence=0.5,
                positive_ratio=0.33,
                negative_ratio=0.33,
                neutral_ratio=0.34,
                article_count=0,
                key_themes=[],
                reasoning="No recent news available for analysis."
            )
        
        # Count sentiments
        positive_count = sum(1 for a in articles if a.sentiment == 'POSITIVE')
        negative_count = sum(1 for a in articles if a.sentiment == 'NEGATIVE')
        neutral_count = sum(1 for a in articles if a.sentiment == 'NEUTRAL')
        total = len(articles)
        
        # Calculate weighted average score (more recent = higher weight)
        articles_sorted = sorted(articles, key=lambda x: x.published_at, reverse=True)
        weighted_scores = []
        for i, article in enumerate(articles_sorted):
            weight = 1.0 / (1 + i * 0.1)  # Decay weight for older articles
            weighted_scores.append(article.sentiment_score * weight * article.relevance)
        
        avg_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        
        # Determine overall sentiment
        if avg_score > 0.15:
            overall = 'POSITIVE'
        elif avg_score < -0.15:
            overall = 'NEGATIVE'
        else:
            overall = 'NEUTRAL'
        
        # Calculate confidence based on agreement
        max_ratio = max(positive_count, negative_count, neutral_count) / total
        confidence = 0.5 + (max_ratio - 0.33) * 0.5  # Scale from 0.5 to 0.83
        
        # Extract key themes (simplified - production uses topic modeling)
        all_text = ' '.join([a.headline for a in articles])
        key_themes = self._extract_themes(all_text)
        
        # Generate reasoning
        reasoning = self._generate_sentiment_reasoning(
            overall, avg_score, positive_count, negative_count, neutral_count, total
        )
        
        return SentimentResult(
            overall_sentiment=overall,
            sentiment_score=avg_score,
            confidence=confidence,
            positive_ratio=positive_count / total,
            negative_ratio=negative_count / total,
            neutral_ratio=neutral_count / total,
            article_count=total,
            key_themes=key_themes,
            reasoning=reasoning
        )
    
    def _extract_themes(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key themes from text (simplified version).
        
        Production uses more sophisticated topic modeling.
        """
        # Common stock-related themes to look for
        theme_keywords = {
            'earnings': ['earnings', 'profit', 'revenue', 'quarterly', 'annual'],
            'growth': ['growth', 'expansion', 'increase', 'gain'],
            'regulation': ['regulation', 'legal', 'compliance', 'lawsuit', 'investigation'],
            'technology': ['technology', 'ai', 'innovation', 'product', 'launch'],
            'market': ['market', 'sector', 'industry', 'competition'],
            'management': ['ceo', 'executive', 'leadership', 'management'],
            'dividend': ['dividend', 'buyback', 'shareholder'],
            'merger': ['merger', 'acquisition', 'deal', 'partnership']
        }
        
        text_lower = text.lower()
        theme_scores = {}
        
        for theme, keywords in theme_keywords.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                theme_scores[theme] = score
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, _ in sorted_themes[:top_n]]
    
    def _generate_sentiment_reasoning(
        self,
        overall: str,
        score: float,
        positive: int,
        negative: int,
        neutral: int,
        total: int
    ) -> str:
        """Generate human-readable sentiment reasoning."""
        sentiment_desc = {
            'POSITIVE': 'bullish',
            'NEGATIVE': 'bearish',
            'NEUTRAL': 'mixed'
        }
        
        return (
            f"News sentiment is {sentiment_desc[overall]} based on {total} recent articles. "
            f"Analysis shows {positive} positive, {negative} negative, and {neutral} neutral articles. "
            f"Overall sentiment score: {score:.2f} on a scale of -1 (very bearish) to +1 (very bullish)."
        )
    
    def get_signal(
        self,
        symbol: str
    ) -> Dict:
        """
        Get sentiment-based trading signal for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with signal and metadata
        """
        # Fetch and analyze news
        articles = self.fetch_news(symbol)
        sentiment = self.aggregate_sentiment(articles)
        
        # Determine signal
        if sentiment.sentiment_score > 0.2 and sentiment.confidence > 0.6:
            signal = 'BULLISH'
        elif sentiment.sentiment_score < -0.2 and sentiment.confidence > 0.6:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'signal': signal,
            'sentiment': sentiment.overall_sentiment,
            'score': sentiment.sentiment_score,
            'confidence': sentiment.confidence,
            'article_count': sentiment.article_count,
            'positive_ratio': sentiment.positive_ratio,
            'negative_ratio': sentiment.negative_ratio,
            'neutral_ratio': sentiment.neutral_ratio,
            'key_themes': sentiment.key_themes,
            'reasoning': sentiment.reasoning
        }


# Example usage
if __name__ == '__main__':
    print("Sentiment Analyzer Demo")
    print("=" * 50)
    
    # Initialize analyzer (without API keys for demo)
    analyzer = SentimentAnalyzer()
    
    # Demo text analysis
    sample_headlines = [
        "Apple stock surges after strong quarterly earnings beat expectations",
        "Tech stocks plunge amid market uncertainty and recession fears",
        "Microsoft announces new partnership, shares rise modestly",
        "Tesla faces investigation over safety concerns",
        "NVIDIA reports record revenue driven by AI demand"
    ]
    
    print("\nAnalyzing sample headlines:\n")
    for headline in sample_headlines:
        sentiment, score = analyzer.analyze_text(headline)
        print(f"Headline: {headline}")
        print(f"  Sentiment: {sentiment} (score: {score:.2f})")
        print()
    
    # Demo aggregate sentiment
    from datetime import datetime
    sample_articles = [
        NewsArticle(
            source="Reuters",
            headline=headline,
            description=None,
            url=None,
            published_at=datetime.now(),
            sentiment=analyzer.analyze_text(headline)[0],
            sentiment_score=analyzer.analyze_text(headline)[1],
            relevance=1.0
        )
        for headline in sample_headlines
    ]
    
    result = analyzer.aggregate_sentiment(sample_articles)
    
    print("\nAggregate Sentiment Result:")
    print(f"  Overall: {result.overall_sentiment}")
    print(f"  Score: {result.sentiment_score:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Positive: {result.positive_ratio:.1%}")
    print(f"  Negative: {result.negative_ratio:.1%}")
    print(f"  Neutral: {result.neutral_ratio:.1%}")
    print(f"  Key Themes: {', '.join(result.key_themes)}")
    print(f"\nReasoning: {result.reasoning}")
