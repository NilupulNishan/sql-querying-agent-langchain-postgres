-- ============================================================
-- db_setup.sql
-- Run this in pgAdmin 4 → Query Tool on your database
-- This creates the sunglasses store tables and seeds sample data
-- ============================================================

-- Drop tables if they exist (for clean re-runs)
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS inventory;

-- ============================================================
-- Table 1: inventory
-- ============================================================
CREATE TABLE inventory (
    item_id             VARCHAR(10)     PRIMARY KEY,
    name                VARCHAR(50)     NOT NULL,
    description         TEXT,
    quantity_in_stock   INTEGER         NOT NULL DEFAULT 0,
    price               NUMERIC(10, 2)  NOT NULL
);

-- ============================================================
-- Table 2: transactions
-- ============================================================
CREATE TABLE transactions (
    transaction_id              VARCHAR(10)     PRIMARY KEY,
    customer_name               VARCHAR(100)    NOT NULL,
    transaction_summary         TEXT,
    transaction_amount          NUMERIC(10, 2)  NOT NULL,
    balance_after_transaction   NUMERIC(10, 2)  NOT NULL,
    created_at                  TIMESTAMP       DEFAULT NOW()
);

-- ============================================================
-- Seed: inventory data (same 6 items as the notebook)
-- ============================================================
INSERT INTO inventory (item_id, name, description, quantity_in_stock, price) VALUES
(
    'SG001', 'Aviator',
    'Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.',
    23, 80.00
),
(
    'SG002', 'Wayfarer',
    'Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.',
    6, 95.00
),
(
    'SG003', 'Mystique',
    'Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.',
    3, 70.00
),
(
    'SG004', 'Sport',
    'Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.',
    11, 110.00
),
(
    'SG005', 'Classic',
    'Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.',
    10, 60.00
),
(
    'SG006', 'Moon',
    'Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.',
    10, 120.00
);

-- ============================================================
-- Seed: transactions — opening balance only
-- ============================================================
INSERT INTO transactions (transaction_id, customer_name, transaction_summary, transaction_amount, balance_after_transaction) VALUES
(
    'TXN001', 'OPENING_BALANCE',
    'Daily opening register balance',
    500.00, 500.00
);

-- ============================================================
-- Verify data was inserted correctly
-- ============================================================
SELECT 'inventory rows: ' || COUNT(*)::TEXT AS check_result FROM inventory
UNION ALL
SELECT 'transactions rows: ' || COUNT(*)::TEXT FROM transactions;