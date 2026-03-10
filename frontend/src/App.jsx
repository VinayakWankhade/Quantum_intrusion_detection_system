import React, { useState, useEffect, useRef } from 'react';
import { Shield, Activity, HardDrive, Cpu, AlertTriangle, CheckCircle, Radio, Network, Zap } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [trafficStream, setTrafficStream] = useState([]);
  const [stats, setStats] = useState({
    totalPackets: 0,
    attacksBlocked: 0,
    avgLatency: 0,
  });

  const [chartData, setChartData] = useState([]);
  const [attackDistribution, setAttackDistribution] = useState([
    { name: 'Benign', value: 1, color: '#10B981' },
    { name: 'Attack', value: 0, color: '#EF4444' }
  ]);

  const ws = useRef(null);
  const terminalEndRef = useRef(null);
  const maxConsoleLines = 50;

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trafficStream]);

  useEffect(() => {
    const initData = Array.from({ length: 25 }, (_, i) => ({
      time: i,
      volume: 0,
      attacks: 0
    }));
    setChartData(initData);

    const connectWebSocket = () => {
      ws.current = new WebSocket('ws://localhost:8000/ws/traffic');

      ws.current.onopen = () => setIsConnected(true);

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.error) return;

        setTrafficStream(prev => {
          const newStream = [...prev, { ...data, id: Date.now() + Math.random() }];
          return newStream.length > maxConsoleLines ? newStream.slice(newStream.length - maxConsoleLines) : newStream;
        });

        setStats(prev => ({
          totalPackets: prev.totalPackets + 1,
          attacksBlocked: prev.attacksBlocked + (data.predicted === 1 ? 1 : 0),
          avgLatency: prev.totalPackets === 0
            ? data.latency_ms
            : ((prev.avgLatency * prev.totalPackets) + data.latency_ms) / (prev.totalPackets + 1)
        }));

        setAttackDistribution(prev => {
          const isAttack = data.predicted === 1;
          return [
            { ...prev[0], value: prev[0].value + (isAttack ? 0 : 1) },
            { ...prev[1], value: prev[1].value + (isAttack ? 1 : 0) }
          ];
        });

        setChartData(prev => {
          const newData = [...prev.slice(1)];
          const lastPoint = { ...newData[newData.length - 1] };
          newData.push({
            time: new Date(data.timestamp).toLocaleTimeString([], { hour12: false, second: '2-digit', minute: '2-digit' }),
            volume: lastPoint.volume * 0.5 + (data.src_bytes / 50) + Math.random() * 20,
            attacks: data.predicted === 1 ? 80 : 0
          });
          return newData;
        });
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };
    };

    connectWebSocket();
    return () => ws.current && ws.current.close();
  }, []);

  return (
    <div className="min-h-screen p-4 md:p-6 lg:p-8 flex flex-col items-center">
      <div className="w-full max-w-7xl">
        {/* Header */}
        <header className="flex justify-between items-center mb-8 relative z-10">
          <div className="flex items-center gap-4">
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-brand to-accent rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-500"></div>
              <div className="relative p-3.5 bg-surface/80 backdrop-blur-md rounded-xl border border-white/10">
                <Shield className="w-7 h-7 text-brand" />
              </div>
            </div>
            <div>
              <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">Nexus<span className="text-brand font-light">Guard</span></h1>
              <p className="text-sm text-gray-400 mt-0.5 tracking-wide flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-accent" /> Quantum-Enhanced Threat Intel
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2.5 px-5 py-2.5 rounded-full border backdrop-blur-md transition-all duration-500 ${isConnected ? 'bg-success/10 border-success/20 shadow-[0_0_15px_rgba(16,185,129,0.15)]' : 'bg-danger/10 border-danger/20 shadow-[0_0_15px_rgba(239,68,68,0.15)]'}`}>
              <Radio className={`w-4 h-4 ${isConnected ? 'text-success animate-pulse' : 'text-danger'}`} />
              <span className={`text-sm font-bold tracking-wide ${isConnected ? 'text-successglow-text text-success' : 'text-danger'}`}>
                {isConnected ? 'NODE SECURE' : 'UPLINK LOST'}
              </span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 relative z-10">

          {/* Main Content Area */}
          <div className="xl:col-span-8 flex flex-col gap-8">

            {/* Top KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                { title: "Processed Packets", value: stats.totalPackets.toLocaleString(), icon: <Activity className="w-4 h-4 text-brand" />, color: "brand" },
                { title: "Anomalies Neutralized", value: stats.attacksBlocked.toLocaleString(), icon: <AlertTriangle className="w-4 h-4 text-danger" />, color: "danger", highlight: true },
                { title: "Engine Latency", value: `${stats.avgLatency.toFixed(1)}ms`, icon: <Cpu className="w-4 h-4 text-success" />, color: "success" }
              ].map((kpi, idx) => (
                <div key={idx} className={`glass-panel p-6 flex flex-col justify-center group ${kpi.highlight ? 'border-danger/30' : ''}`}>
                  <div className={`absolute -right-6 -top-6 w-32 h-32 bg-${kpi.color}/10 rounded-full blur-3xl transition-opacity group-hover:bg-${kpi.color}/20`}></div>
                  <div className="flex items-center gap-2 mb-2">
                    {kpi.icon}
                    <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest">{kpi.title}</p>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl font-extrabold font-mono tracking-tight ${kpi.highlight ? 'text-danger glow-text' : 'text-white'}`}>
                      {kpi.value}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Traffic Volume Chart */}
            <div className="glass-panel p-6 flex-1 min-h-[350px] flex flex-col">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-bold flex items-center gap-2 tracking-wide">
                  <Network className="w-5 h-5 text-accent" />
                  Telemetry Visualization
                </h2>
                <div className="flex gap-4">
                  <div className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-accent shadow-[0_0_8px_rgba(59,130,246,0.8)]"></div><span className="text-xs text-gray-400 font-medium tracking-wider">VOLUME</span></div>
                  <div className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-danger shadow-[0_0_8px_rgba(239,68,68,0.8)]"></div><span className="text-xs text-gray-400 font-medium tracking-wider">THREATS</span></div>
                </div>
              </div>
              <div className="flex-1 w-full relative -ml-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorAttacks" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#EF4444" stopOpacity={0.6} />
                        <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="time" stroke="#6B7280" fontSize={11} tickLine={false} axisLine={false} dy={10} />
                    <YAxis stroke="#6B7280" fontSize={11} tickLine={false} axisLine={false} dx={-10} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(18, 22, 38, 0.9)', backdropFilter: 'blur(8px)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }}
                      itemStyle={{ color: '#F3F4F6', fontWeight: 600 }}
                    />
                    <Area type="monotone" dataKey="volume" stroke="#3B82F6" strokeWidth={3} fill="url(#colorVolume)" activeDot={{ r: 6, fill: '#3B82F6', stroke: '#fff', strokeWidth: 2 }} />
                    <Area type="step" dataKey="attacks" stroke="#EF4444" strokeWidth={2} fill="url(#colorAttacks)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Model Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-panel p-6">
                <h3 className="text-xs font-bold text-gray-400 mb-5 flex items-center gap-2 uppercase tracking-widest border-b border-white/10 pb-3">
                  <Cpu className="w-4 h-4 text-gray-300" /> Classical Processing
                </h3>
                <div className="space-y-3">
                  <div className="glass-card">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold text-gray-200">Random Forest</span>
                      <span className="text-xs font-mono bg-white/5 px-2 py-1 rounded text-gray-300 border border-white/5">Active</span>
                    </div>
                    <div className="flex items-end gap-6">
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Acc</p>
                        <p className="text-lg font-mono text-white">99.8%</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Latency</p>
                        <p className="text-lg font-mono text-brand">1.2ms</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="glass-panel p-6 relative">
                <div className="absolute top-0 right-0 w-48 h-48 bg-brand/10 blur-3xl rounded-full pointer-events-none"></div>
                <h3 className="text-xs font-bold text-brand mb-5 flex items-center gap-2 uppercase tracking-widest border-b border-brand/20 pb-3">
                  <Zap className="w-4 h-4" /> Quantum Processing
                </h3>
                <div className="space-y-3">
                  <div className="glass-card border-brand/20 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-brand"></div>
                    <div className="flex justify-between items-center mb-2 pl-2">
                      <span className="font-semibold text-gray-200">QSVM<span className="text-gray-500 font-normal text-xs ml-2">ZZFeatureMap</span></span>
                      <span className="text-[10px] uppercase font-bold text-brand bg-brand/10 px-2 py-1 rounded">Simulated</span>
                    </div>
                    <div className="flex items-end gap-6 pl-2">
                      <div>
                        <p className="text-[10px] text-brand/60 uppercase tracking-wider mb-1">Acc</p>
                        <p className="text-lg font-mono text-white">94.1%</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Terminal & Pie */}
          <div className="xl:col-span-4 flex flex-col gap-8 h-full">

            {/* Attack Distribution Donut */}
            <div className="glass-panel p-6 h-[350px] flex flex-col">
              <h2 className="text-[11px] font-bold text-gray-400 mb-2 uppercase tracking-widest text-center">Threat Distribution</h2>
              <div className="flex-1 w-full relative">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={attackDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={80}
                      outerRadius={100}
                      paddingAngle={8}
                      dataKey="value"
                      stroke="none"
                      cornerRadius={6}
                    >
                      {attackDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} style={{ filter: `drop-shadow(0px 0px 8px ${entry.color}80)` }} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(18, 22, 38, 0.9)', backdropFilter: 'blur(8px)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px' }}
                      itemStyle={{ color: '#F3F4F6', fontWeight: 600 }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex items-center justify-center flex-col pointer-events-none">
                  <span className="text-4xl font-extrabold font-mono text-white glow-text">
                    {stats.totalPackets > 0 ? Math.round((stats.attacksBlocked / stats.totalPackets) * 100) : 0}<span className="text-2xl text-gray-400">%</span>
                  </span>
                  <span className="text-xs text-gray-400 font-medium tracking-widest mt-1">HOSTILE</span>
                </div>
              </div>
            </div>

            {/* Live Terminal Log */}
            <div className="glass-panel flex-1 flex flex-col overflow-hidden min-h-[450px]">
              <div className="px-5 py-4 border-b border-white/5 bg-surfaceHighlight/50 flex justify-between items-center z-10">
                <h2 className="text-[11px] font-bold flex items-center gap-2 uppercase tracking-widest text-gray-300">
                  <HardDrive className="w-4 h-4 text-accent" />
                  Live Triage Console
                </h2>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-gray-600/50"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-warning/50"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-brand shadow-[0_0_8px_rgba(96,165,250,0.8)] animate-ping-slow"></div>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-2.5 bg-black/40 terminal-text relative">
                {trafficStream.length === 0 && (
                  <div className="text-gray-500 flex items-center justify-center h-full animate-pulse tracking-widest text-xs uppercase">Initializing socket connection...</div>
                )}
                {trafficStream.map((p) => (
                  <div key={p.id} className={`p-3 rounded-lg border flex flex-col gap-1.5 animate-slide-in backdrop-blur-sm ${p.predicted === 1
                      ? 'bg-danger/10 border-danger/30 text-danger/90 threat-detected'
                      : 'bg-white/[0.03] border-white/5 text-gray-400 hover:bg-white/[0.05]'
                    }`}>
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-gray-500 font-mono">[{new Date(p.timestamp).toLocaleTimeString()}]</span>
                        <span className="font-mono text-[13px] text-gray-200">
                          SEQ: {p.src_bytes}B <span className="text-gray-600 mx-1">➜</span> {p.dst_bytes}B
                        </span>
                      </div>
                      {p.predicted === 1 ? (
                        <span className="flex items-center gap-1.5 font-bold text-danger bg-danger/20 border border-danger/30 px-2 py-1 rounded text-[10px] tracking-wider select-none shadow-[0_0_10px_rgba(239,68,68,0.3)]">
                          <AlertTriangle className="w-3.5 h-3.5" /> CRITICAL
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5 text-success bg-success/10 border border-success/20 px-2 py-1 rounded text-[10px] tracking-wider select-none">
                          <CheckCircle className="w-3 h-3" /> SAFE
                        </span>
                      )}
                    </div>
                    <div className="flex justify-between items-center mt-1 border-t border-white/5 pt-2">
                      <span className="text-[10px] font-mono text-gray-500 flex items-center gap-2">
                        <span className="px-1.5 py-0.5 rounded bg-white/5 text-gray-300">TCP {p.protocol_type}</span>
                        {p.model_used}
                      </span>
                      <span className="text-[10px] font-mono font-bold whitespace-nowrap">
                        {p.latency_ms.toFixed(1)}ms | <span className={p.confidence_percent > 90 ? 'text-white' : 'text-gray-500'}>{p.confidence_percent}%</span>
                      </span>
                    </div>
                  </div>
                ))}
                <div ref={terminalEndRef} />
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
