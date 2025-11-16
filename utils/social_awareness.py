import numpy as np
import jieba
import re
from collections import Counter
import nltk
import spacy
import warnings

warnings.filterwarnings('ignore')


class SocialAwarenessAnalyzer:
    """Social and emotional awareness analysis tool"""

    def __init__(self):
        # Initialize social bias indicators
        self.bias_indicators = {
            'gender': ['male', 'female', 'man', 'woman', 'gentleman', 'lady', 'husband', 'wife', 'father', 'mother',
                       'dad', 'mom', 'boy', 'girl', 'son', 'daughter'],
            'region': ['northern', 'southern', 'outsider', 'rural', 'urban', 'countryside', 'city', 'mountainous',
                       'coastal', 'east', 'west', 'north', 'south', 'inland', 'border', 'urban-rural', 'city person',
                       'villager'],
            'economic': ['rich', 'poor', 'wealthy', 'affluent', 'money', 'broke', 'high-income', 'low-income',
                         'poverty', 'prosperity', 'middle-class', 'white-collar', 'blue-collar', 'worker', 'salary',
                         'class', 'social class'],
            'age': ['young', 'elderly', 'child', 'old', 'youth', 'middle-aged', 'senior', 'kid', 'teenager', 'adult',
                    'gray-haired', 'wrinkles', 'youthful', 'vitality', 'late-life', 'aging', 'childlike', 'mature'],
            'profession': ['doctor', 'lawyer', 'teacher', 'worker', 'farmer', 'student', 'civil servant',
                           'businessperson', 'entrepreneur', 'migrant worker', 'white-collar', 'executive', 'expert',
                           'professor', 'researcher', 'technician']
        }

        # Cultural sensitivity indicators
        self.cultural_sensitivity = {
            'ethnic_terms': ['ethnic minority', 'Han', 'Hui', 'Tibetan', 'Uyghur', 'Mongolian', 'Miao', 'Yi', 'Zhuang',
                             'Korean', 'ethnic group', 'tribe', 'race', 'indigenous'],
            'religious_terms': ['Buddhism', 'Taoism', 'Islam', 'Christianity', 'Catholicism', 'temple', 'church',
                                'mosque', 'scripture', 'prayer', 'worship', 'faith', 'religion', 'deity', 'God',
                                'Buddha'],
            'political_terms': ['government', 'policy', 'regulation', 'party', 'country', 'leader', 'constitution',
                                'law', 'election', 'democracy', 'human rights', 'sovereignty', 'territory', 'politics',
                                'system', 'regime', 'official'],
            'historical_events': ['war of resistance', 'cultural revolution', 'reform and opening', 'party founding',
                                  'nation founding', 'historical event', 'cultural heritage', 'traditional custom',
                                  'history', 'revolution', 'war', 'colonialism', 'liberation', 'unification']
        }

        # Emotional safety indicators
        self.safety_concerns = {
            'violence': ['hit', 'kill', 'harm', 'violence', 'attack', 'fight', 'conflict', 'retaliation', 'threat',
                         'intimidation', 'beat', 'murder', 'self-harm', 'suicide', 'weapon', 'danger', 'terror',
                         'assault'],
            'discrimination': ['discrimination', 'prejudice', 'exclusion', 'humiliation', 'degradation', 'insult',
                               'slander', 'contempt', 'mockery', 'sarcasm', 'derision', 'denigration', 'discriminatory',
                               'insulting', 'xenophobia', 'hate', 'hostility'],
            'illegal': ['illegal', 'crime', 'unlawful', 'steal', 'rob', 'fraud', 'traffic', 'drug', 'gambling',
                        'pornography', 'black market', 'corruption', 'bribery', 'money laundering', 'scam', 'smuggling',
                        'counterfeit', 'infringement']
        }

        # Emotional cues
        self.emotional_cues = {
            'positive': ['satisfied', 'excellent', 'good', 'great', 'quality', 'recommend', 'like', 'appreciate',
                         'thank', 'perfect', 'surprise', 'happy', 'joyful', 'praise', 'relief', 'pride', 'joy',
                         'contentment'],
            'negative': ['disappointed', 'dissatisfied', 'terrible', 'bad', 'trash', 'useless', 'problem', 'failure',
                         'defect', 'complaint', 'anger', 'frustration', 'worry', 'anxiety', 'pain', 'depression',
                         'sadness', 'despair'],
            'neutral': ['normal', 'average', 'standard', 'routine', 'ordinary', 'typical', 'common', 'regular',
                        'nothing special', 'neutral', 'objective', 'fact', 'data', 'statistics', 'report', 'analysis',
                        'research']
        }

        # Social interaction patterns
        self.interaction_patterns = {
            'question': ['?', 'how', 'what', 'why', 'who', 'where', 'when', 'whether', 'can', 'may', 'could', 'should'],
            'command': ['please', 'should', 'must', 'need', 'suggest', 'request', 'order', 'forbid', 'prohibit',
                        'required', 'certainly', 'hope', 'expect', 'appeal'],
            'emotion': ['happy', 'joyful', 'satisfied', 'disappointed', 'angry', 'worried', 'like', 'dislike', 'moved',
                        'sad', 'excited', 'surprised', 'fear', 'disgust', 'shame', 'guilt'],
            'fact': ['data shows', 'report indicates', 'according to', 'research shows', 'statistics', 'result',
                     'situation', 'fact', 'evidence', 'proof', 'confirm', 'verify', 'record', 'document', 'literature',
                     'history']
        }

        # Social contexts
        self.social_contexts = {
            'public_affairs': ['government', 'policy', 'regulation', 'society', 'public', 'country', 'ethnicity',
                               'community', 'city', 'village', 'development', 'reform', 'planning', 'construction'],
            'business': ['enterprise', 'company', 'product', 'service', 'price', 'quality', 'customer', 'market',
                         'business', 'economy', 'industry', 'sector', 'investment', 'profit', 'sales', 'brand'],
            'personal': ['I', 'personal', 'family', 'life', 'experience', 'feeling', 'experience', 'story', 'memory',
                         'emotion', 'mood', 'health', 'growth', 'learning', 'work'],
            'professional': ['professional', 'technology', 'science', 'medicine', 'law', 'education', 'research',
                             'academic', 'expert', 'scholar', 'knowledge', 'skill', 'training', 'degree', 'certificate',
                             'qualification']
        }

    def analyze_social_bias(self, text):
        """Analyze social bias in text"""
        results = {}
        total_words = len(text.split())

        if total_words == 0:
            return results

        # Analyze each category of bias
        for category, indicators in self.bias_indicators.items():
            count = sum(1 for word in indicators if word.lower() in text.lower())
            density = count / total_words
            examples = [word for word in indicators if word.lower() in text.lower()][:3]  # First 3 examples

            results[category] = {
                'count': count,
                'density': density,
                'examples': examples,
                'level': self._get_bias_level(density)
            }

        # Calculate overall bias score
        overall_bias = sum(v['density'] for v in results.values()) / len(results) if results else 0
        results['overall_bias_score'] = overall_bias
        results['overall_bias_level'] = self._get_bias_level(overall_bias)

        return results

    def analyze_cultural_sensitivity(self, text):
        """Analyze cultural sensitivity"""
        results = {}
        total_words = len(text.split())

        if total_words == 0:
            return results

        sensitivity_score = 0
        detected_terms = []
        context_factors = 0

        # Detect culturally sensitive terms
        for category, terms in self.cultural_sensitivity.items():
            count = sum(1 for term in terms if term.lower() in text.lower())
            if count > 0:
                detected_terms.extend([term for term in terms if term.lower() in text.lower()])
                sensitivity_score += count * 0.3  # Increase score by 0.3 per sensitive term
                context_factors += 1

        # Adjust score based on context
        negative_context = ['conflict', 'opposition', 'contradiction', 'controversy', 'dispute', 'discrimination',
                            'exclusion', 'hostility', 'hate', 'attack', 'degradation', 'contempt']
        positive_context = ['harmony', 'unity', 'co-development', 'mutual respect', 'inclusion', 'understanding',
                            'equality', 'respect', 'diversity', 'coexistence', 'tolerance']

        if any(term.lower() in text.lower() for term in negative_context):
            sensitivity_score += 0.5
            context_factors += 1

        if any(term.lower() in text.lower() for term in positive_context):
            sensitivity_score -= 0.3
            context_factors += 1

        # Normalize score
        if context_factors > 0:
            sensitivity_score = sensitivity_score / context_factors

        # Limit to 0-1 range
        sensitivity_score = max(0, min(1.0, sensitivity_score))

        # Get examples
        examples = detected_terms[:5] if detected_terms else []

        return {
            'score': sensitivity_score,
            'detected_terms': detected_terms,
            'examples': examples,
            'level': self._get_sensitivity_level(sensitivity_score),
            'context_adjusted': True if context_factors > 0 else False
        }

    def analyze_emotional_safety(self, text):
        """Analyze emotional safety"""
        results = {}
        total_words = len(text.split())

        if total_words == 0:
            return results

        safety_score = 1.0  # 1.0 represents safest
        detected_terms = []

        # Detect safety concern terms
        for category, terms in self.safety_concerns.items():
            count = sum(1 for term in terms if term.lower() in text.lower())
            if count > 0:
                detected_terms.extend([term for term in terms if term.lower() in text.lower()])
                # Reduce score by 0.2 per safety concern term
                safety_score -= count * 0.2

        # Adjust based on overall tone
        negative_tone = ['unfortunate', 'tragedy', 'disaster', 'pain', 'despair', 'helplessness', 'danger', 'threat',
                         'panic', 'fear', 'hatred', 'hostility']
        positive_tone = ['hope', 'positive', 'optimistic', 'solution', 'help', 'support', 'care', 'understanding',
                         'tolerance', 'constructive', 'cooperation', 'peace']

        if any(term.lower() in text.lower() for term in negative_tone):
            safety_score -= 0.3

        if any(term.lower() in text.lower() for term in positive_tone):
            safety_score += 0.2

        # Limit to 0-1 range
        safety_score = max(0, min(1.0, safety_score))

        # Get examples
        examples = detected_terms[:5] if detected_terms else []

        return {
            'score': safety_score,
            'detected_terms': detected_terms,
            'examples': examples,
            'level': self._get_safety_level(safety_score),
            'recommendations': self._get_safety_recommendations(safety_score, detected_terms)
        }

    def analyze_emotional_cues(self, text, sentiment_probs):
        """Analyze emotional cues and social interaction patterns"""
        # Emotional consistency
        emotional_consistency = self._assess_emotional_consistency(text, sentiment_probs)

        # Social interaction patterns
        interaction_patterns = self._analyze_interaction_patterns(text)

        # Social context
        social_context = self._assess_social_context(text)

        return {
            'emotional_consistency': emotional_consistency,
            'interaction_patterns': interaction_patterns,
            'social_context': social_context,
            'detailed_factors': self._get_detailed_emotional_factors(text)
        }

    def _get_bias_level(self, density):
        """Get bias level based on density"""
        if density > 0.05:
            return "High"
        elif density > 0.02:
            return "Medium"
        else:
            return "Low"

    def _get_sensitivity_level(self, score):
        """Get sensitivity level based on score"""
        if score > 0.7:
            return "Highly Sensitive"
        elif score > 0.4:
            return "Moderately Sensitive"
        else:
            return "Low Sensitivity"

    def _get_safety_level(self, score):
        """Get safety level based on score"""
        if score > 0.8:
            return "Safe"
        elif score > 0.5:
            return "Moderate"
        else:
            return "Requires Attention"

    def _get_safety_recommendations(self, score, detected_terms):
        """Get safety recommendations"""
        if score > 0.8:
            return ["Content is safe, no modifications needed"]
        elif score > 0.5:
            return ["Consider reducing sensitive terminology", "Use more neutral expressions",
                    "Add constructive content"]
        else:
            terms = ", ".join(detected_terms[:3]) if detected_terms else "sensitive terms"
            return [
                f"High-risk content detected: {terms}",
                "Consider rewriting content to avoid inflammatory language",
                "Add positive guidance and solution-oriented content",
                "Consider impact on diverse audiences"
            ]

    def _assess_emotional_consistency(self, text, sentiment_probs):
        """Assess emotional consistency"""
        negative_prob = sentiment_probs[0]
        neutral_prob = sentiment_probs[1]
        positive_prob = sentiment_probs[2]

        # Identify emotional words
        negative_words = self.emotional_cues['negative']
        positive_words = self.emotional_cues['positive']

        negative_count = sum(1 for word in negative_words if word.lower() in text.lower())
        positive_count = sum(1 for word in positive_words if word.lower() in text.lower())

        # Assess consistency
        if negative_prob > 0.6 and negative_count > positive_count:
            return "High"  # Emotionally consistent
        elif positive_prob > 0.6 and positive_count > negative_count:
            return "High"
        elif neutral_prob > 0.6 and abs(negative_count - positive_count) < 2:
            return "High"
        elif abs(negative_prob - positive_prob) < 0.2 and negative_count > 0 and positive_count > 0:
            return "Medium"  # Mixed emotions
        else:
            return "Low"  # Emotionally inconsistent

    def _analyze_interaction_patterns(self, text):
        """Analyze interaction patterns"""
        patterns = []

        # Detect patterns
        for pattern_type, indicators in self.interaction_patterns.items():
            if any(term.lower() in text.lower() for term in indicators):
                patterns.append(pattern_type)

        return patterns if patterns else ["Neutral"]

    def _assess_social_context(self, text):
        """Assess social context"""
        contexts = []

        # Detect context
        for context_type, indicators in self.social_contexts.items():
            if any(term.lower() in text.lower() for term in indicators):
                contexts.append(context_type)

        return contexts if contexts else ["General"]

    def _get_detailed_emotional_factors(self, text):
        """Get detailed emotional factors"""
        factors = {
            'emotional_words': [],
            'tone_indicators': [],
            'social_signals': []
        }

        # Emotional words
        for category, words in self.emotional_cues.items():
            found = [word for word in words if word.lower() in text.lower()]
            if found:
                factors['emotional_words'].extend([(category, word) for word in found])

        # Tone indicators
        tone_indicators = {
            'positive': ['very', 'extremely', 'absolutely', 'completely', 'so', 'really', 'super', 'huge', 'immensely'],
            'negative': ['not', 'no', 'none', 'lack', 'very bad', 'terrible', 'serious', 'extremely', 'excessive'],
            'neutral': ['maybe', 'perhaps', 'or', 'seems', 'basically', 'generally', 'relatively', 'quite',
                        'to some extent']
        }

        for category, words in tone_indicators.items():
            found = [word for word in words if word.lower() in text.lower()]
            if found:
                factors['tone_indicators'].extend([(category, word) for word in found])

        # Social signals
        social_signals = {
            'respect': ['respect', 'respectful', 'honor', 'value', 'treasure', 'equal treatment',
                        'careful consideration'],
            'empathy': ['understanding', 'empathy', 'shared feeling', 'consideration', 'care', 'emotional resonance',
                        'perspective-taking'],
            'inclusion': ['include', 'inclusion', 'acceptance', 'welcome', 'unity', 'together', 'we', 'community']
        }

        for category, words in social_signals.items():
            found = [word for word in words if word.lower() in text.lower()]
            if found:
                factors['social_signals'].extend([(category, word) for word in found])

        return factors