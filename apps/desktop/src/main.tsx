import React from "react";
import ReactDOM from "react-dom/client";

import "./styles/app.css";
import { App } from "./App";
import { initMedia } from "./lib/ipc";

void initMedia().finally(() => ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
));
