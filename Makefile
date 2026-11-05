.PHONY: help deps-backend deps-frontend test test-backend test-frontend build-frontend run-dev smoke smoke-ui smoke-ui-stack smoke-all qa ci-e2e ci-full ci ci-core clean

PYTHON ?= python3
NPM ?= npm
SMOKE_STACK_API_PORT ?= 8010
SMOKE_STACK_FRONT_PORT ?= 3010

help:
	@echo "Kullanılabilir hedefler:"
	@echo "  help             - Bu yardım listesini gösterir"
	@echo "  deps-backend     - Python bağımlılıklarını yükler (requirements.txt)"
	@echo "  deps-frontend    - Next.js bağımlılıklarını yükler"
	@echo "  test-backend     - Backend birim testlerini koşar"
	@echo "  test-frontend    - Frontend build doğrulaması (npm run build)"
	@echo "  build-frontend   - Frontend üretim build'i"
	@echo "  run-dev          - run_dev.py ile backend+frontend'i başlatır"
	@echo "  smoke            - run_dev orkestrasyonu smoke testini çalıştırır"
	@echo "  smoke-ui         - Playwright ile temel UI etkileşim smoke testi (opsiyonel)"
	@echo "  smoke-ui-stack   - Backend+frontend'i başlatıp Playwright UI smoke testini otomatik koşar"
	@echo "  smoke-all        - make smoke ve make smoke-ui-stack hedeflerini sırayla koşar"
	@echo "  qa               - make ci-e2e (günlük kapanışa yakın, UI e2e dahil)"
	@echo "  ci-e2e           - make ci-core + make smoke-ui-stack (CI + tarayıcı e2e)"
	@echo "  ci-full          - make ci-e2e (tam akış: bağımlılıklar+test+build+backend ve e2e smoke)"
	@echo "  ci               - Local CI akışı: bağımlılıklar + test + build + smoke"
	@echo "  clean            - Geçici build dosyalarını temizler"

deps-backend:
	$(PYTHON) -m pip install -r requirements.txt

deps-frontend:
	cd frontend && $(NPM) install

test-backend:
	$(PYTHON) -m unittest discover -s backend/tests -p 'test_*.py'

test-frontend:
	cd frontend && $(NPM) run build

build-frontend:
	cd frontend && $(NPM) run build

run-dev:
	$(PYTHON) run_dev.py --install-frontend-deps

smoke:
	$(PYTHON) scripts/run_dev_smoke.py

smoke-ui:
	@node scripts/run_frontend_ui_smoke.mjs

smoke-ui-stack:
	$(PYTHON) scripts/run_frontend_ui_smoke_with_stack.py \
		--api-port $(SMOKE_STACK_API_PORT) \
		--front-port $(SMOKE_STACK_FRONT_PORT)

smoke-all:
	$(MAKE) smoke
	$(MAKE) smoke-ui-stack SMOKE_STACK_API_PORT=8020 SMOKE_STACK_FRONT_PORT=3020

qa:
	$(MAKE) ci-e2e

ci-core:
	$(MAKE) deps-backend
	$(MAKE) deps-frontend
	$(MAKE) test-backend
	$(MAKE) test-frontend

ci-e2e:
	$(MAKE) ci-core
	$(MAKE) smoke-ui-stack SMOKE_STACK_API_PORT=8020 SMOKE_STACK_FRONT_PORT=3020

ci-full:
	$(MAKE) ci-e2e

ci:
	$(MAKE) ci-core
	$(MAKE) smoke

test: test-backend

clean:
	- rm -rf frontend/.next
	- find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
