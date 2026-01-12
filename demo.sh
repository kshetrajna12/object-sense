uv run object-sense reset-world -f
uv run object-sense ingest -v /home/kshetrajna/demos/object-sense/notes/leopard_sighting_pompom.md 2>&1 | grep -E "OK|entities"
uv run object-sense ingest -v /home/kshetrajna/demos/object-sense/images/01_C_resting.jpg 2>&1 | grep -E "OK|entities"
uv run object-sense ingest -v /home/kshetrajna/demos/object-sense/images/02_B_standing.jpg 2>&1 | grep -E "OK|entities"
uv run object-sense ingest -v /home/kshetrajna/demos/object-sense/images/03_A_feeding.jpg 2>&1 | grep -E "OK|entities"
uv run object-sense ingest -v /home/kshetrajna/demos/object-sense/images/04_D_other.jpg 2>&1 | grep -E "OK|entities"
