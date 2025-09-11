# Contributing to streamLens Analytics

Hey there! Thanks for checking out streamLens Analytics. I'm Cazandra, and this is a project I've been working on to help make media representation more transparent and fair through data analysis.

## What This Project Is About

I started building streamLens Analytics because I noticed there wasn't enough transparency around representation in streaming media. My goal is to create tools that can:

- Actually measure representation gaps in movies and shows on streaming platforms
- Give content creators and platforms real data they can use to make better decisions
- Help promote more inclusive storytelling by making the numbers visible

Basically, I want to turn "we need better representation" into "here's exactly where we stand and how we can improve."

## Ways You Can Help

### Found a Bug or Have an Idea?

If something's broken or you think of a feature that would be useful:
- **Bug Reports**: Tell me what you were trying to do, what you expected to happen, and what actually happened
- **Feature Requests**: Explain what you'd like to see and why it would be helpful
- **Data Problems**: Let me know if you spot errors or gaps in the data

### Want to Contribute Code?

I'd love your help! Here's how to get started:

#### Setting Up Your Development Environment
```bash
# Fork and clone the repository
git clone https://github.com/CazandraAporbo/streamlens-analytics.git
cd streamlens-analytics

# Create a virtual environment (this keeps dependencies organized)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install what you need for development
pip install -r requirements-dev.txt

# Set up pre-commit hooks (these check your code automatically)
pre-commit install
```

#### My Workflow for Changes
I like to keep things organized, so here's what works best:

1. Create a branch for your feature: `git checkout -b feature/describe-what-youre-adding`
2. Make your changes (following the coding style below)
3. Write or update tests: `pytest tests/`
4. Check code formatting: `black . && flake8 .`
5. Commit with a clear message explaining what you did
6. Push your branch and create a Pull Request

I'll review it and we can discuss any changes needed.

### Have Data to Share?

This is huge! The project really benefits from good data. I'm especially looking for:

- **Subtitle/Script Data**: Any properly licensed dialogue data you can share
- **Cast Demographics**: Verified information about who's in what shows/movies
- **Platform Information**: Details about content on different streaming services
- **Manual Annotations**: If you've done any bias detection work, that's gold

#### How to Format Data Submissions
```json
{
  "content_id": "unique_identifier",
  "title": "Content Title",
  "platform": "netflix", 
  "year": 2024,
  "characters": [
    {
      "name": "Character Name",
      "actor": "Actor Name",
      "demographics": {
        "gender": "category",
        "race": "category", 
        "age_group": "category"
      },
      "screen_time_minutes": 45.5,
      "dialogue_word_count": 2340
    }
  ]
}
```

### Research and Analysis

If you're doing research or analysis with streamLens Analytics, I'd love to hear about it:
- **Academic Papers**: Research using the tools
- **Blog Posts**: Your findings and what they mean
- **New Visualizations**: Different ways to show the data
- **Better Methods**: Improvements to how we analyze things

## Coding Style (Keep It Clean)

### Python Code
I try to write code that explains itself. Here's what good code looks like in this project:

```python
# Good: Clear function with type hints and documentation
def calculate_representation_score(
    demographics: Dict[str, int],
    baseline: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate how well represented different groups are.
    
    Args:
        demographics: How many people are in each group
        baseline: What we'd expect to see (defaults to equal distribution)
        
    Returns:
        Score from 0-1 (higher = more representative)
    """
    # Implementation here
    pass

# Not so good: Unclear and no explanation
def calc_score(d, b=None):
    # What does this do? Who knows!
    pass
```

### JavaScript Code
Same philosophy for the frontend:

```javascript
/**
 * Update the chart when new data comes in
 * @param {Object} data - The processed media data
 * @param {string} chartType - What kind of chart to show
 */
function updateVisualization(data, chartType) {
    // Make sure we have what we need
    if (!data || !chartType) {
        console.error('Missing data or chart type');
        return;
    }
    
    // Update based on chart type
    switch(chartType) {
        case 'bubble':
            updateBubbleChart(data);
            break;
        case 'bar':
            updateBarChart(data);
            break;
        default:
            console.warn(`Unknown chart type: ${chartType}`);
    }
}
```

### Commit Messages That Make Sense
```
feat(analyzer): add diversity index calculation

This implements Shannon diversity index to measure how evenly
distributed different demographic groups are in a piece of content.

Fixes #23
```

Use these prefixes: `feat` (new feature), `fix` (bug fix), `docs` (documentation), `style` (formatting), `refactor` (code cleanup), `test` (adding tests), `chore` (maintenance)

## Testing (Because Bugs Are Annoying)

### Unit Tests
I write tests for individual functions to make sure they work:

```python
# tests/test_analyzer.py
def test_diversity_index():
    """Make sure diversity index calculation works correctly"""
    analyzer = RepresentationAnalyzer()
    
    # Test when everything is evenly distributed
    demographics = ['A', 'B', 'C', 'A', 'B', 'C']
    result = analyzer.calculate_diversity_index(demographics)
    assert result == pytest.approx(1.0)  # Should be perfectly diverse
    
    # Test when one group dominates
    demographics = ['A', 'A', 'A', 'B']  
    result = analyzer.calculate_diversity_index(demographics)
    assert result < 0.7  # Should be less diverse
```

### Integration Tests
These test that different parts work together:

```python
def test_full_analysis_pipeline():
    """Test the whole process from start to finish"""
    processor = DataProcessor()
    test_data = processor.load_test_data()
    results = processor.process_data(test_data)
    
    # Make sure we get the expected results
    assert 'overall_metrics' in results
    assert results['overall_metrics']['diversity_index'] > 0
    assert len(results['character_analysis']) > 0
```

## Security (Keeping Things Safe)

A few important rules:
- Never commit API keys, passwords, or any secrets to the repository
- Always validate user inputs if you're working on the web interface
- Use parameterized queries for any database operations
- If you find a security issue, please reach out to me directly at [your-email]@becaziam.com

## Documentation (Help Others Understand)

### Document Your Code
Every function should explain what it does:

```python
def analyze_representation(
    content_id: str,
    metrics: List[str] = ['diversity', 'parity'],
    include_temporal: bool = True
) -> AnalysisResult:
    """
    Analyze how well different groups are represented in a piece of content.
    
    This function calculates various metrics to understand representation
    patterns in movies or TV shows.
    
    Args:
        content_id: Unique identifier for the content (like "netflix_stranger_things_s1")
        metrics: Which metrics to calculate:
            - 'diversity': Shannon diversity index (how evenly distributed groups are)
            - 'parity': Gender parity score (how balanced male/female representation is)  
            - 'sentiment': Analyzes the sentiment of dialogue by different groups
        include_temporal: Whether to include trends over time
        
    Returns:
        AnalysisResult containing:
            - metrics: Dictionary with calculated scores
            - visualizations: Chart configurations for displaying results
            - insights: Key findings in plain English
            
    Raises:
        ContentNotFoundError: When the content_id doesn't exist in our database
        InvalidMetricError: When you request a metric we don't support
        
    Example:
        >>> result = analyze_representation(
        ...     content_id="netflix_stranger_things_s1",
        ...     metrics=['diversity', 'parity']
        ... )
        >>> print(f"Diversity score: {result.metrics['diversity']}")
        Diversity score: 0.847
    """
```

## Design Guidelines (Making It Look Good)

### Keep Accessibility in Mind
- Follow WCAG 2.1 AA guidelines so everyone can use the tools
- Make sure visualizations are understandable without explanation
- Design mobile-first since people use phones for everything
- Use lazy loading for large datasets so pages load quickly

### Color Usage
```css
/* Use variables so we can change themes easily */
.positive-trend {
    color: var(--success-color);  /* Good approach */
}

.negative-trend {
    color: #ff0000;  /* Avoid hardcoding colors like this */
}
```

## Code Review Process

### When You're Reviewing
- Actually run the code locally to test it
- Think about edge cases that might break things
- Check that documentation is complete and accurate
- Make sure tests pass and cover the new functionality
- Consider how changes might affect performance

### Review Checklist
- [ ] Code follows the style guidelines above
- [ ] Tests are added or updated appropriately
- [ ] Documentation explains what changed
- [ ] No obvious security problems
- [ ] Performance impact seems reasonable
- [ ] Accessibility isn't broken

## Data Ethics (Being Responsible)

### Core Principles
This is really important to me:

1. **Privacy**: We never collect or use personal viewer data
2. **Accuracy**: Demographic information should be verified and respectful
3. **Representation**: Use inclusive categories that don't exclude people
4. **Transparency**: Always document where data comes from and what its limitations are

### Avoiding Bias
- Acknowledge when our datasets have limitations or gaps
- Avoid analysis that could reinforce harmful stereotypes
- Include diverse perspectives when interpreting results
- Regularly audit our own tools for bias

## Release Process (Getting Changes Live)

When we're ready to release a new version:

1. Update the version number in `setup.py`
2. Update `CHANGELOG.md` with what changed
3. Run the full test suite to make sure nothing broke
4. Build updated documentation
5. Tag the release: `git tag -a v1.0.0 -m "Release version 1.0.0"`
6. Deploy to production

## Getting Help

### Where to Ask Questions
- **GitHub Issues**: Best for bug reports and feature requests
- **GitHub Discussions**: Good for general questions and brainstorming
- **Pull Requests**: For proposing code changes

### Community Guidelines
- Be respectful and constructive in all interactions
- Follow the code of conduct
- Help others when you can
- Share your knowledge and insights

## Recognition

If you contribute to the project, you'll be recognized in:
- The CONTRIBUTORS.md file
- Project README
- Release notes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions or want to discuss something:
- Open a GitHub Discussion for general questions
- Create a GitHub Issue for specific problems or suggestions
- Email me directly at [your-email]@becaziam.com for anything sensitive

Thanks for considering contributing to streamLens Analytics! Every bit of help makes the project better and gets us closer to real transparency in media representation.

---

**Building a more inclusive media landscape through data**