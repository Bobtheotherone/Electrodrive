import type { CSSProperties, ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";

/**
 * App shell for day-one sections (Design Doc "What ResearchED must ship with on day one"):
 * - Run Library, Run Launcher, Live Monitor, Post-run Dashboards, Comparison, Experimental Upgrades.
 *
 * Design Doc: §3.1 architecture; §2.2 non-goal: GUI is optional extra.
 * Repo: Bobtheotherone/Electrodrive keeps ResearchED backend deps optional (electrodrive/researched/__init__.py).
 */

const shell: CSSProperties = {
  fontFamily:
    'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"',
  color: "#111827",
  background: "#ffffff",
  minHeight: "100vh",
  display: "flex",
  flexDirection: "column",
};

const topbar: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "12px 16px",
  borderBottom: "1px solid #e5e7eb",
};

const brand: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 10,
};

const nav: CSSProperties = {
  display: "flex",
  gap: 10,
  flexWrap: "wrap",
};

const main: CSSProperties = {
  padding: "16px",
  flex: 1,
};

function NavItem(props: { to: string; children: ReactNode }) {
  return (
    <NavLink
      to={props.to}
      style={({ isActive }) => ({
        display: "inline-block",
        padding: "6px 10px",
        borderRadius: 8,
        textDecoration: "none",
        color: isActive ? "#111827" : "#374151",
        background: isActive ? "#e5e7eb" : "transparent",
        border: "1px solid #e5e7eb",
      })}
    >
      {props.children}
    </NavLink>
  );
}

export default function App() {
  return (
    <div style={shell}>
      <header style={topbar}>
        <div style={brand}>
          <div style={{ fontSize: 18, fontWeight: 700 }}>ResearchED</div>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Local runs • Live monitor • Dashboards
          </div>
        </div>
        <nav style={nav} aria-label="Primary">
          <NavItem to="/runs">Runs</NavItem>
          <NavItem to="/launch">Launch</NavItem>
          <NavItem to="/compare">Compare</NavItem>
          <NavItem to="/upgrades">Upgrades</NavItem>
        </nav>
      </header>

      <main style={main}>
        <Outlet />
      </main>
    </div>
  );
}
