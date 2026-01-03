"""E2E test for slot normalization in the entity resolution pipeline."""

import pytest
from uuid import uuid4

from object_sense.inference.schemas import DeterministicId, EntityHypothesis, SlotValue
from object_sense.models import EntityNature, LinkStatus, Medium
from object_sense.resolution.resolver import EntityResolver


@pytest.mark.integration
class TestSlotNormalizationE2E:
    """Test slot normalization integrates with entity resolution."""

    async def test_entity_created_with_normalized_slots(
        self,
        db_session,
        make_observation,
        make_signature,
        caplog,
    ):
        """Entity slots are normalized during creation via entity_seed."""
        import logging
        caplog.set_level(logging.WARNING)

        # Create observation WITHOUT deterministic_ids at observation level
        # This way the entity is created from the entity_seed with slots
        obs = make_observation(
            medium=Medium.JSON,
            observation_kind="product_record",
            deterministic_ids=[],  # Empty - let entity_seed handle it
        )
        sig = make_signature(observation_id=obs.observation_id)
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        # Create entity seed with slots
        det_id = DeterministicId(
            id_type="sku",
            id_value="SLOT-TEST-001",
            id_namespace="test",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="product",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Test Product With Slots",
            deterministic_ids=[det_id],
            confidence=0.9,
            # These slots use the SlotValue schema
            slots=[
                SlotValue(name="price", value="42.99", is_reference=False),
                SlotValue(name="species", value="leopard", is_reference=False),  # Should warn
                SlotValue(name="category", value="wildlife", is_reference=False),  # Should warn
            ],
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        # Verify entity created
        assert len(result.entities_created) == 1
        entity = result.entities_created[0]
        
        print("\n=== Slot Normalization E2E Test ===")
        print(f"Entity created: {entity.entity_id}")
        print(f"Entity slots: {entity.slots}")
        
        # Slots should be present and normalized
        slots = entity.slots
        assert "price" in slots
        assert "species" in slots
        assert "category" in slots
        
        # Check that slots are normalized (wrapped in the expected format)
        # The _slots_to_dict creates: {"value": <val>} for non-references
        # Then normalize_slots wraps this as {"value": <val>, "type": "string"}
        price_slot = slots["price"]
        print(f"Price slot structure: {price_slot}")
        
        # After normalization, should have wrapped structure with "type"
        assert "value" in price_slot
        assert "type" in price_slot
        
        # Check warnings were logged for entity-like slots
        warnings = [r for r in caplog.records if "Entity slot hygiene" in r.message]
        print(f"Slot hygiene warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w.message}")
        
        # species and category should trigger warnings about entity references
        warning_messages = " ".join(w.message for w in warnings)
        assert "species" in warning_messages or "category" in warning_messages

    async def test_entity_slots_normalized_with_type(
        self,
        db_session,
        make_observation,
        make_signature,
    ):
        """Entity slots have correct normalized structure with type field."""
        obs = make_observation(
            medium=Medium.JSON,
            observation_kind="test",
            deterministic_ids=[],
        )
        sig = make_signature(observation_id=obs.observation_id)
        db_session.add(obs)
        db_session.add(sig)
        await db_session.flush()

        det_id = DeterministicId(
            id_type="test_id",
            id_value="NORM-TEST-001",
            id_namespace="test",
            strength="strong",
        )
        entity_seed = EntityHypothesis(
            entity_type="test",
            entity_nature=EntityNature.INDIVIDUAL,
            suggested_name="Normalization Test",
            deterministic_ids=[det_id],
            confidence=0.9,
            slots=[
                SlotValue(name="description", value="A test product", is_reference=False),
                SlotValue(name="quantity", value="42", is_reference=False),
            ],
        )

        resolver = EntityResolver(db_session)
        result = await resolver.resolve(
            observation=obs,
            signatures=[sig],
            entity_seeds=[entity_seed],
        )

        entity = result.entities_created[0]
        print(f"\n=== Slot Normalization Structure Test ===")
        print(f"Entity slots: {entity.slots}")
        
        # Verify normalized structure
        assert "description" in entity.slots
        assert "quantity" in entity.slots
        
        desc_slot = entity.slots["description"]
        print(f"Description slot: {desc_slot}")
        
        # Should have normalized structure with type
        assert "type" in desc_slot, f"Expected 'type' in slot, got: {desc_slot}"
        assert desc_slot["type"] == "string"
        assert desc_slot["value"] == "A test product"


@pytest.mark.integration  
class TestSlotNormalizationDirect:
    """Direct tests of slot normalization without full pipeline."""
    
    def test_normalize_slots_function_directly(self):
        """Test normalize_slots function directly."""
        from object_sense.utils.slots import normalize_slots
        
        raw_slots = {
            "price": 42.99,
            "name": "Test Product",
            "active": True,
            "species": "leopard",  # Should warn about entity reference
        }
        
        normalized, warnings = normalize_slots(raw_slots)
        
        print("\n=== Direct normalize_slots Test ===")
        print(f"Input: {raw_slots}")
        print(f"Output: {normalized}")
        print(f"Warnings: {warnings}")
        
        # Check normalized structure
        assert normalized["price"]["type"] == "number"
        assert normalized["price"]["value"] == 42.99
        
        assert normalized["name"]["type"] == "string"
        assert normalized["name"]["value"] == "Test Product"
        
        assert normalized["active"]["type"] == "boolean"
        assert normalized["active"]["value"] is True
        
        # Check warnings
        assert len(warnings) >= 2  # At least price (no unit) and species (entity-like)
        
        # species should have a warning about being better as entity reference
        species_warning = [w for w in warnings if "species" in w]
        assert len(species_warning) > 0
        
    def test_extract_slot_for_index(self):
        """Test extract_slot_for_index extracts the right value."""
        from object_sense.utils.slots import extract_slot_for_index
        
        # Primitive wrapper
        result = extract_slot_for_index({"value": 42.99, "type": "number", "unit": "USD"})
        assert result == 42.99
        
        # Reference
        entity_id = str(uuid4())
        result = extract_slot_for_index({"ref_entity_id": entity_id, "ref_type": "species"})
        assert result == entity_id
        
        # List
        result = extract_slot_for_index([
            {"value": "a", "type": "string"},
            {"value": "b", "type": "string"},
        ])
        assert result == ["a", "b"]
        
        print("\n=== extract_slot_for_index Test ===")
        print("All extractions working correctly!")
