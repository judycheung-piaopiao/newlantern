
# Experiments

## Baseline
Simple rule: only the first previous exam is marked as relevant, others as not relevant.

## What worked
- API returns all required predictions and passes the public contract check.

## What failed
- No use of exam details, so accuracy is limited.

## Next steps
- Use exam time or metadata to improve relevance.

## What I can improve
- Add logic to use exam time, type, or findings for better predictions.
- Try simple ML models if more data is available.
