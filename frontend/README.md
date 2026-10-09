# Frontend

Next.js + TypeScript frontend scaffold for the DR screening API.

## Run

1. npm install
2. npm run dev

Set API base URL in .env.local:

NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

## CI/CD

The GitHub Actions workflow runs frontend linting, TypeScript checks, and a
production build for pull requests and pushes to `main`. Successful pushes to
`main` publish the frontend container to GitHub Container Registry as
`ghcr.io/<owner>/<repository>/frontend:latest` and a commit SHA tag.

Set the repository Actions variable `NEXT_PUBLIC_API_BASE_URL` to the public API
URL before publishing. If it is unset, the image uses `http://localhost:8000`.
