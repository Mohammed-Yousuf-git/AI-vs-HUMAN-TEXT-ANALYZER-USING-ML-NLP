import React, { useState } from 'react';
import { Brain, User, Sparkles, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface PredictionResult {
  prediction: 'AI' | 'Human';
  confidence: number;
}

interface HumanizeResult {
  rewritten_text: string;
}

function App() {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [humanizedText, setHumanizedText] = useState<string>('');
  const [isDetecting, setIsDetecting] = useState(false);
  const [isHumanizing, setIsHumanizing] = useState(false);
  const [error, setError] = useState<string>('');

  const handleDetect = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }

    setIsDetecting(true);
    setError('');
    setPrediction(null);
    setHumanizedText('');

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze text');
      }

      const result: PredictionResult = await response.json();
      setPrediction(result);
    } catch (err) {
      setError('Error analyzing text. Please make sure the backend server is running.');
      console.error('Detection error:', err);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleHumanize = async () => {
    if (!inputText.trim()) return;

    setIsHumanizing(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/humanize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error('Failed to humanize text');
      }

      const result: HumanizeResult = await response.json();
      setHumanizedText(result.rewritten_text);
    } catch (err) {
      setError('Error humanizing text. Please try again.');
      console.error('Humanization error:', err);
    } finally {
      setIsHumanizing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBackground = (confidence: number) => {
    if (confidence >= 80) return 'bg-green-100 border-green-300';
    if (confidence >= 60) return 'bg-yellow-100 border-yellow-300';
    return 'bg-red-100 border-red-300';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full blur-lg opacity-20 scale-110"></div>
              <div className="relative bg-gradient-to-r from-indigo-600 to-purple-600 p-4 rounded-full">
                <Brain className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
            AI vs Human Text Detector
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Analyze text to determine if it's AI-generated or human-written, then humanize AI content with advanced rewriting technology.
          </p>
        </div>

        {/* Main Interface */}
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 border border-white/50">
          {/* Text Input */}
          <div className="mb-8">
            <label htmlFor="text-input" className="block text-sm font-semibold text-gray-700 mb-3">
              Enter text to analyze
            </label>
            <textarea
              id="text-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste or type the text you want to analyze..."
              className="w-full h-40 p-4 border border-gray-300 rounded-2xl focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all duration-300 resize-none bg-white/80 backdrop-blur-sm"
              disabled={isDetecting || isHumanizing}
            />
            <div className="mt-2 text-sm text-gray-500">
              {inputText.length} characters
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-2xl flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <span className="text-red-700">{error}</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4 mb-8">
            <button
              onClick={handleDetect}
              disabled={isDetecting || isHumanizing || !inputText.trim()}
              className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-4 px-8 rounded-2xl font-semibold text-lg transition-all duration-300 hover:from-indigo-700 hover:to-purple-700 hover:shadow-lg hover:scale-105 disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            >
              {isDetecting ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Brain className="h-5 w-5" />
                  Detect AI vs Human
                </>
              )}
            </button>

            {prediction && prediction.prediction === 'AI' && (
              <button
                onClick={handleHumanize}
                disabled={isHumanizing || isDetecting || !inputText.trim()}
                className="flex-1 bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-4 px-8 rounded-2xl font-semibold text-lg transition-all duration-300 hover:from-emerald-600 hover:to-teal-700 hover:shadow-lg hover:scale-105 disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                {isHumanizing ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Humanizing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" />
                    Humanize Text
                  </>
                )}
              </button>
            )}
          </div>

          {/* Prediction Results */}
          {prediction && (
            <div className={`mb-8 p-6 rounded-2xl border-2 transition-all duration-500 ${getConfidenceBackground(prediction.confidence)}`}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  {prediction.prediction === 'AI' ? (
                    <Brain className="h-6 w-6 text-indigo-600" />
                  ) : (
                    <User className="h-6 w-6 text-emerald-600" />
                  )}
                  <h3 className="text-xl font-bold text-gray-800">
                    Prediction: {prediction.prediction === 'AI' ? 'AI-Generated' : 'Human-Written'}
                  </h3>
                </div>
                <CheckCircle2 className="h-6 w-6 text-green-600" />
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-700 font-medium">Confidence Score:</span>
                  <span className={`font-bold text-lg ${getConfidenceColor(prediction.confidence)}`}>
                    {prediction.confidence.toFixed(1)}%
                  </span>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-1000 ${
                      prediction.confidence >= 80 ? 'bg-green-500' :
                      prediction.confidence >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${prediction.confidence}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}

          {/* Humanized Result */}
          {humanizedText && (
            <div className="bg-gradient-to-r from-emerald-50 to-teal-50 p-6 rounded-2xl border border-emerald-200">
              <div className="flex items-center gap-3 mb-4">
                <Sparkles className="h-6 w-6 text-emerald-600" />
                <h3 className="text-xl font-bold text-gray-800">Humanized Version</h3>
              </div>
              <div className="bg-white/70 backdrop-blur-sm p-4 rounded-xl border border-emerald-200">
                <p className="text-gray-800 leading-relaxed">{humanizedText}</p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-gray-600">
          <p className="text-sm">
            Powered by machine learning and natural language processing
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;