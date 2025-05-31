#!/usr/bin/env python3
"""
Anti-Duplication Discovery Script

This script automates the discovery phase of the anti-duplication workflow,
searching for existing functionality before implementing new features.
"""

import os
import re
import subprocess
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

class AntiDuplicationDiscovery:
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'query': '',
            'semantic_results': [],
            'file_results': [],
            'content_results': [],
            'assessments': [],
            'recommendation': None
        }
    
    def semantic_search(self, query: str) -> List[Dict]:
        """
        Perform semantic search for concepts and functionality.
        This would integrate with VS Code's semantic search capabilities.
        """
        print(f"🔍 Semantic Search: {query}")
        
        # Common semantic patterns for different types of functionality
        semantic_patterns = {
            'deployment': ['deploy', 'publish', 'release', 'push', 'sync'],
            'automation': ['script', 'batch', 'auto', 'pipeline', 'workflow'],
            'configuration': ['config', 'setup', 'init', 'settings', 'env'],
            'data_processing': ['process', 'transform', 'parse', 'convert', 'etl'],
            'api': ['api', 'service', 'client', 'request', 'endpoint'],
            'ui': ['component', 'widget', 'interface', 'form', 'display']
        }
        
        # Find relevant patterns
        query_lower = query.lower()
        relevant_patterns = []
        for category, patterns in semantic_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                relevant_patterns.extend(patterns)
        
        # Search for files containing semantic patterns
        results = []
        search_extensions = ['.py', '.js', '.ts', '.sh', '.md', '.yml', '.yaml', '.json']
        
        for ext in search_extensions:
            for pattern in relevant_patterns:
                try:
                    cmd = f"find {self.workspace_root} -name '*{pattern}*{ext}' -type f"
                    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                    files = [f.strip() for f in output.strip().split('\n') if f.strip()]
                    
                    for file_path in files:
                        results.append({
                            'type': 'semantic_file',
                            'path': file_path,
                            'pattern': pattern,
                            'category': self._categorize_file(file_path),
                            'confidence': self._calculate_semantic_confidence(file_path, query)
                        })
                except subprocess.CalledProcessError:
                    continue
        
        # Remove duplicates and sort by confidence
        unique_results = {r['path']: r for r in results}.values()
        sorted_results = sorted(unique_results, key=lambda x: x['confidence'], reverse=True)
        
        self.results['semantic_results'] = sorted_results[:10]  # Top 10 results
        return sorted_results[:10]
    
    def file_pattern_search(self, patterns: List[str]) -> List[Dict]:
        """Search for files matching specific patterns."""
        print(f"📁 File Pattern Search: {patterns}")
        
        results = []
        for pattern in patterns:
            try:
                cmd = f"find {self.workspace_root} -path '*{pattern}*' -type f"
                output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                files = [f.strip() for f in output.strip().split('\n') if f.strip()]
                
                for file_path in files:
                    results.append({
                        'type': 'pattern_match',
                        'path': file_path,
                        'pattern': pattern,
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        'modified': os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                    })
            except subprocess.CalledProcessError:
                continue
        
        # Sort by modification time (most recent first)
        sorted_results = sorted(results, key=lambda x: x['modified'], reverse=True)
        self.results['file_results'] = sorted_results
        return sorted_results
    
    def content_search(self, search_terms: List[str]) -> List[Dict]:
        """Search for specific content within files."""
        print(f"📝 Content Search: {search_terms}")
        
        results = []
        search_extensions = ['.py', '.js', '.ts', '.sh', '.md', '.yml', '.yaml']
        
        for term in search_terms:
            try:
                # Build grep command for multiple file types
                include_args = ' '.join([f"--include='*{ext}'" for ext in search_extensions])
                cmd = f"grep -r -n -i '{term}' {self.workspace_root} {include_args}"
                output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                
                for line in output.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path, line_num, content = parts[0], parts[1], parts[2]
                            results.append({
                                'type': 'content_match',
                                'path': file_path,
                                'line': int(line_num),
                                'content': content.strip(),
                                'term': term,
                                'relevance': self._calculate_content_relevance(content, term)
                            })
            except subprocess.CalledProcessError:
                continue
        
        # Sort by relevance
        sorted_results = sorted(results, key=lambda x: x['relevance'], reverse=True)
        self.results['content_results'] = sorted_results[:20]  # Top 20 results
        return sorted_results[:20]
    
    def assess_file(self, file_path: str, query: str) -> Dict:
        """Assess a discovered file for relevance and quality."""
        print(f"📊 Assessing: {file_path}")
        
        if not os.path.exists(file_path):
            return {'error': 'File not found'}
        
        # Basic file analysis
        file_size = os.path.getsize(file_path)
        file_ext = Path(file_path).suffix
        
        # Read file content (first 1000 lines to avoid huge files)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:1000]
                content = ''.join(lines)
        except Exception as e:
            return {'error': f'Cannot read file: {str(e)}'}
        
        # Assessment scores
        assessment = {
            'path': file_path,
            'size': file_size,
            'lines': len(lines),
            'extension': file_ext,
            'scores': {
                'feature_coverage': self._assess_feature_coverage(content, query),
                'code_quality': self._assess_code_quality(content, file_ext),
                'documentation': self._assess_documentation(content),
                'integration_effort': self._assess_integration_effort(content, file_ext),
                'maintainability': self._assess_maintainability(content, file_ext)
            },
            'analysis': {
                'purpose': self._extract_purpose(content),
                'dependencies': self._extract_dependencies(content, file_ext),
                'api_surface': self._extract_api_surface(content, file_ext),
                'complexity': self._assess_complexity(content)
            }
        }
        
        # Calculate overall score
        scores = assessment['scores']
        assessment['overall_score'] = sum(scores.values()) / len(scores)
        assessment['recommendation'] = self._generate_recommendation(assessment['overall_score'])
        
        return assessment
    
    def _categorize_file(self, file_path: str) -> str:
        """Categorize file based on path and extension."""
        path_lower = file_path.lower()
        
        if '/script' in path_lower or '/bin' in path_lower:
            return 'script'
        elif '/doc' in path_lower or '/readme' in path_lower:
            return 'documentation'
        elif '/test' in path_lower or 'test_' in path_lower:
            return 'test'
        elif '/config' in path_lower or '/setup' in path_lower:
            return 'configuration'
        elif '.py' in path_lower:
            return 'python'
        elif '.js' in path_lower or '.ts' in path_lower:
            return 'javascript'
        elif '.sh' in path_lower:
            return 'shell'
        else:
            return 'other'
    
    def _calculate_semantic_confidence(self, file_path: str, query: str) -> float:
        """Calculate confidence score for semantic match."""
        path_lower = file_path.lower()
        query_lower = query.lower()
        
        score = 0.0
        query_words = query_lower.split()
        
        # Path relevance
        for word in query_words:
            if word in path_lower:
                score += 0.3
        
        # File category relevance
        category = self._categorize_file(file_path)
        if category in ['script', 'python'] and any(word in query_lower for word in ['script', 'automation', 'tool']):
            score += 0.2
        
        # Recency bonus (newer files are more relevant)
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            days_old = (datetime.now().timestamp() - mtime) / (24 * 3600)
            if days_old < 30:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_content_relevance(self, content: str, term: str) -> float:
        """Calculate relevance score for content match."""
        content_lower = content.lower()
        term_lower = term.lower()
        
        score = 0.0
        
        # Exact match bonus
        if term_lower in content_lower:
            score += 0.5
        
        # Context relevance (function/class definition)
        if any(keyword in content_lower for keyword in ['def ', 'class ', 'function ', 'const ']):
            score += 0.3
        
        # Comment relevance
        if any(char in content for char in ['#', '//', '/*']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_feature_coverage(self, content: str, query: str) -> int:
        """Assess how well the file covers the requested feature (1-5)."""
        query_words = query.lower().split()
        content_lower = content.lower()
        
        coverage_count = sum(1 for word in query_words if word in content_lower)
        coverage_ratio = coverage_count / len(query_words) if query_words else 0
        
        return max(1, min(5, int(coverage_ratio * 5) + 1))
    
    def _assess_code_quality(self, content: str, file_ext: str) -> int:
        """Assess code quality (1-5)."""
        if file_ext not in ['.py', '.js', '.ts', '.sh']:
            return 3  # Neutral for non-code files
        
        score = 3  # Start with neutral
        
        # Positive indicators
        if 'def ' in content or 'function ' in content:
            score += 1
        if '"""' in content or '/*' in content:  # Documentation strings
            score += 1
        if 'import ' in content or 'require(' in content:  # Proper imports
            score += 0.5
        
        # Negative indicators
        if 'TODO' in content or 'FIXME' in content:
            score -= 0.5
        if len(content.split('\n')) > 500:  # Very long files
            score -= 0.5
        
        return max(1, min(5, int(score)))
    
    def _assess_documentation(self, content: str) -> int:
        """Assess documentation quality (1-5)."""
        doc_indicators = ['"""', "'''", '/*', '#', 'README', 'USAGE', 'EXAMPLE']
        doc_count = sum(1 for indicator in doc_indicators if indicator in content)
        
        # Check for structured documentation
        if any(header in content for header in ['##', '===', '---']):
            doc_count += 1
        
        return max(1, min(5, doc_count))
    
    def _assess_integration_effort(self, content: str, file_ext: str) -> int:
        """Assess how easy it would be to integrate (1-5, higher is easier)."""
        if file_ext == '.md':
            return 5  # Documentation is easy to integrate
        
        score = 3  # Start neutral
        
        # Easy integration indicators
        if 'if __name__ == "__main__"' in content:
            score += 1  # Standalone script
        if 'argparse' in content or 'sys.argv' in content:
            score += 1  # CLI interface
        
        # Difficult integration indicators
        if 'import ' in content:
            import_count = content.count('import ')
            if import_count > 10:
                score -= 1  # Many dependencies
        
        return max(1, min(5, score))
    
    def _assess_maintainability(self, content: str, file_ext: str) -> int:
        """Assess maintainability (1-5)."""
        if file_ext == '.md':
            return 4  # Documentation is generally maintainable
        
        score = 3  # Start neutral
        
        # Maintainability indicators
        lines = content.split('\n')
        if len(lines) < 200:
            score += 1  # Reasonable size
        if any(word in content.lower() for word in ['test', 'unittest', 'pytest']):
            score += 1  # Has tests
        
        # Anti-patterns
        if 'global ' in content:
            score -= 1  # Global variables
        if len([line for line in lines if len(line) > 120]) > len(lines) * 0.1:
            score -= 1  # Many long lines
        
        return max(1, min(5, score))
    
    def _extract_purpose(self, content: str) -> str:
        """Extract the purpose/description from file content."""
        lines = content.split('\n')
        
        # Look for docstrings, comments, or README content
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                # Multi-line docstring
                purpose = line[3:]
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().endswith('"""') or lines[j].strip().endswith("'''"):
                        purpose += ' ' + lines[j].strip()[:-3]
                        break
                    purpose += ' ' + lines[j].strip()
                return purpose.strip()
            elif line.startswith('#') and len(line) > 10:
                return line[1:].strip()
        
        # Fallback: return first substantial line
        for line in lines[:10]:
            if len(line.strip()) > 20 and not line.strip().startswith(('import', 'from', '#!')):
                return line.strip()
        
        return "Purpose not clearly documented"
    
    def _extract_dependencies(self, content: str, file_ext: str) -> List[str]:
        """Extract dependencies from file content."""
        deps = []
        
        if file_ext == '.py':
            # Python imports
            import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'
            for match in re.finditer(import_pattern, content):
                dep = match.group(1) or match.group(2)
                if dep and not dep.startswith('.'):
                    deps.append(dep.split('.')[0])
        
        elif file_ext in ['.js', '.ts']:
            # JavaScript/TypeScript imports
            import_pattern = r'(?:import.+from\s+["\']([^"\']+)["\']|require\(["\']([^"\']+)["\']\))'
            for match in re.finditer(import_pattern, content):
                dep = match.group(1) or match.group(2)
                if dep and not dep.startswith('.'):
                    deps.append(dep)
        
        elif file_ext == '.sh':
            # Shell dependencies (rough heuristic)
            for line in content.split('\n'):
                if line.strip().startswith(('source ', '. ', 'bash ', 'python ', 'node ')):
                    deps.append(line.strip().split()[0])
        
        return list(set(deps))  # Remove duplicates
    
    def _extract_api_surface(self, content: str, file_ext: str) -> List[str]:
        """Extract public API (functions, classes, etc.)."""
        api = []
        
        if file_ext == '.py':
            # Python functions and classes
            func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
            
            api.extend(re.findall(func_pattern, content))
            api.extend(re.findall(class_pattern, content))
        
        elif file_ext in ['.js', '.ts']:
            # JavaScript/TypeScript functions
            func_pattern = r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=)'
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            
            matches = re.findall(func_pattern, content)
            api.extend([m[0] or m[1] for m in matches])
            api.extend(re.findall(class_pattern, content))
        
        return api
    
    def _assess_complexity(self, content: str) -> str:
        """Assess complexity level."""
        lines = len(content.split('\n'))
        
        if lines < 50:
            return 'low'
        elif lines < 200:
            return 'medium'
        elif lines < 500:
            return 'high'
        else:
            return 'very_high'
    
    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on overall score."""
        if score >= 4.0:
            return 'use_existing'
        elif score >= 3.0:
            return 'enhance_existing'
        elif score >= 2.0:
            return 'significant_modification'
        else:
            return 'create_new'
    
    def generate_report(self) -> Dict:
        """Generate comprehensive discovery report."""
        # Perform assessment on top results
        top_files = set()
        
        # Add top semantic results
        for result in self.results['semantic_results'][:5]:
            top_files.add(result['path'])
        
        # Add top file pattern results
        for result in self.results['file_results'][:3]:
            top_files.add(result['path'])
        
        # Assess each file
        assessments = []
        for file_path in top_files:
            assessment = self.assess_file(file_path, self.results['query'])
            if 'error' not in assessment:
                assessments.append(assessment)
        
        self.results['assessments'] = assessments
        
        # Generate overall recommendation
        if assessments:
            best_assessment = max(assessments, key=lambda x: x['overall_score'])
            self.results['recommendation'] = {
                'action': best_assessment['recommendation'],
                'best_file': best_assessment['path'],
                'score': best_assessment['overall_score'],
                'rationale': self._generate_rationale(best_assessment)
            }
        else:
            self.results['recommendation'] = {
                'action': 'create_new',
                'rationale': 'No suitable existing functionality found'
            }
        
        return self.results
    
    def _generate_rationale(self, assessment: Dict) -> str:
        """Generate human-readable rationale for recommendation."""
        score = assessment['overall_score']
        scores = assessment['scores']
        
        if score >= 4.0:
            return f"High-quality existing solution found. Feature coverage: {scores['feature_coverage']}/5, Code quality: {scores['code_quality']}/5. Use with minimal modification."
        elif score >= 3.0:
            return f"Good existing solution that needs enhancement. Strongest areas: {max(scores, key=scores.get)}. Consider improving weaker aspects."
        elif score >= 2.0:
            return f"Existing solution requires significant modification. Major concerns: {min(scores, key=scores.get)}. Evaluate cost vs. creating new."
        else:
            return f"Existing solution not suitable. Low scores across multiple areas. Recommend creating new implementation."
    
    def save_report(self, output_file: str = None):
        """Save discovery report to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"anti_duplication_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Anti-Duplication Discovery Tool')
    parser.add_argument('query', help='Description of functionality to search for')
    parser.add_argument('--patterns', nargs='*', help='File patterns to search for')
    parser.add_argument('--terms', nargs='*', help='Content terms to search for')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--workspace', help='Workspace root directory')
    
    args = parser.parse_args()
    
    print("🔍 Anti-Duplication Discovery Tool")
    print("=" * 40)
    
    discovery = AntiDuplicationDiscovery(args.workspace)
    discovery.results['query'] = args.query
    
    # Perform discovery
    print(f"\n🎯 Searching for: {args.query}")
    
    # Semantic search
    discovery.semantic_search(args.query)
    
    # File pattern search
    if args.patterns:
        discovery.file_pattern_search(args.patterns)
    else:
        # Auto-generate patterns from query
        auto_patterns = args.query.lower().split()
        discovery.file_pattern_search(auto_patterns)
    
    # Content search
    if args.terms:
        discovery.content_search(args.terms)
    else:
        # Auto-generate terms from query
        auto_terms = args.query.lower().split()
        discovery.content_search(auto_terms)
    
    # Generate report
    print("\n📊 Generating assessment report...")
    report = discovery.generate_report()
    
    # Display summary
    print("\n" + "=" * 40)
    print("📋 DISCOVERY SUMMARY")
    print("=" * 40)
    
    print(f"Semantic matches found: {len(report['semantic_results'])}")
    print(f"File pattern matches: {len(report['file_results'])}")
    print(f"Content matches: {len(report['content_results'])}")
    print(f"Files assessed: {len(report['assessments'])}")
    
    if report['recommendation']:
        rec = report['recommendation']
        print(f"\n🎯 RECOMMENDATION: {rec['action'].upper()}")
        if 'best_file' in rec:
            print(f"Best match: {rec['best_file']} (score: {rec['score']:.1f}/5.0)")
        print(f"Rationale: {rec['rationale']}")
    
    # Save report
    discovery.save_report(args.output)
    
    print(f"\n✅ Discovery complete! Use the report to make informed decisions about implementation approach.")

if __name__ == '__main__':
    main()
