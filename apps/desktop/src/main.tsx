import React from "react";
import ReactDOM from "react-dom/client";

import "./styles/app.css";
import { App } from "./App";
import { installActivityTracking } from "./lib/activity";
import { initMedia } from "./lib/ipc";

installActivityTracking();

void initMedia().finally(() => ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
));
