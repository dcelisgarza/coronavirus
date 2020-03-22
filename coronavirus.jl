using DifferentialEquations
using Interact, Plots
using DataFrames, CSV

function event(x, xi; scale = 1)
    if length(xi) == 1
        return xi[1] < x ? 1 * scale : 0
    else
        return xi[1] < x <= xi[2] ? 1 * scale : 0
    end
end

function μ_effective(μ, I, serious, beds, β)
    return serious * I > beds ? ((serious * I - beds) * (β - 1) / I + 1) * μ : μ
end

function params(df::DataFrame)
    population = df[1, :population]
    cases = df[1, :cases]
    deaths = df[1, :deaths]
    recovered = df[1, :recovered]
    reproductive_num = df[1, :reproductive_num]

    latent_period = df[1, :latent_period]
    t0 = df[1, :t0]
    tf = df[1, :tf]
    tq = convert.(Float64, df[!, :tq])
    q0 = df[1, :q0]
    serious = df[1, :serious]
    beds = df[1, :beds]
    β = df[1, :beta]
    fv = df[1, :f_vulnerable]
    α = df[1, :alpha]

    return population,
        cases,
        deaths,
        recovered,
        reproductive_num,
        latent_period,
        t0,
        tf,
        tq,
        q0,
        serious,
        beds,
        β,
        fv,
        α
end

function setup(df::DataFrame)
    population,
    cases,
    deaths,
    recovered,
    reproductive_num,
    latent_period,
    t0,
    tf,
    tq,
    q0,
    serious,
    beds,
    β,
    fv,
    α = params(df)
    mortality = df[1, :mortality]

    # Initial condtitions
    R₀ = recovered / population
    S₀ = (population - cases) / population
    D₀ = deaths / population
    I₀ = 1 - S₀ - R₀ - D₀
    u₀ = [S₀; I₀; R₀; D₀]

    # Parameters
    γ = 1 / latent_period
    μ = γ * mortality
    λ = reproductive_num * (γ + μ)
    p = [λ; γ; μ; serious; beds; β; tq; q0; event]

    # time
    tspan = (t0; tf)

    return u₀, p, tspan, population
end

function setupv(df::DataFrame)
    population,
    cases,
    deaths,
    recovered,
    reproductive_num,
    latent_period,
    t0,
    tf,
    tq,
    q0,
    serious,
    beds,
    β,
    fv,
    α = params(df)
    mortality = df[!, :mortality]

    # Initial condtitions
    R₀ = recovered / population
    S₀ = (population - cases) / population
    D₀ = deaths / population
    I₀ = 1.0 - S₀ - R₀ - D₀

    Sv₀ = fv * S₀
    Snv₀ = (1 - fv) * S₀
    Iv₀ = fv * I₀
    Inv₀ = (1 - fv) * I₀

    u₀ = [Sv₀; Snv₀; Iv₀; Inv₀; R₀; D₀]

    # Parameters
    γ = 1 / latent_period
    μv = γ * mortality[2]
    μnv = γ * mortality[1]
    μ = μv * fv + μnv * (1 - fv)
    λ = reproductive_num * (γ + μ)
    p = [λ; α; γ; μv; μnv]

    # time
    tspan = (t0; tf)

    return u₀, p, tspan, population, S₀, I₀
end

function sirdv!(du, u, p, t)
    Sv, Snv, Iv, Inv, R, D = u
    λ, α, γ, μv, μnv = p

    λii = λ
    λij = λ * (1 - α)

    f1 = (λii * Iv + λij * Inv)
    f2 = (λii * Inv + λij * Iv)

    du[1] = -f1 * Sv
    du[2] = -f2 * Snv
    du[3] = f1 * Sv - γ * Iv - μv * Iv
    du[4] = f2 * Snv - γ * Inv - μnv * Inv
    du[5] = γ * (Iv + Inv)
    du[6] = μv * Iv + μnv * Inv
end

function sird!(du, u, p, t)
    S, I, R, D = u
    λ, γ, μ = p[1:3]
    du[1] = -λ * S * I
    du[2] = λ * S * I - γ * I - μ * I
    du[3] = γ * I
    du[4] = μ * I
end

function sirdq!(du, u, p, t)
    S, I, R, D = u
    λ, γ, μ = p[1:3]
    tq = p[7:end-2]
    q0 = p[end-1]
    q = p[end](t, tq; scale = q0)
    du[1] = -λ * (1.0 - q) * S * I
    du[2] = λ * (1.0 - q) * S * I - γ * I - μ * I
    du[3] = γ * I
    du[4] = μ * I
end

function sirdqb!(du, u, p, t)
    S, I, R, D = u
    λ, γ, μ, serious, beds, β = p[1:6]
    μ = μ_effective(μ, I, serious, beds, β)
    tq = p[7:end-2]
    q0 = p[end-1]
    q = p[end](t, tq; scale = q0)
    du[1] = -λ * (1.0 - q) * S * I
    du[2] = λ * (1.0 - q) * S * I - γ * I - μ * I
    du[3] = γ * I
    du[4] = μ * I
end

function flatten_curve(q_arr, equations, u₀, p, tspan, population)
    fig1 = plot()
    fig2 = plot()
    mortality = zeros(length(q_arr))
    for i = 1:length(q_arr)
        p[end-1] = q_arr[i]
        prob = ODEProblem(equations, u₀, tspan, p)
        sol = solve(prob,saveat=2.5)
        # plot!(fig1, sol, vars = 2)
        plot!(
            fig1,
            sol[2,:] .*= p[4],
            xaxis = "Time, days",
            yaxis = "Cases that need beds",
            label = "Q = $(p[end-1])",
            lw = 2,
        )
        mortality[i] = sol[4, end] / sol[3, end]
    end
    plot!(
        fig2,
        q_arr,
        mortality,
        xaxis = "Q",
        yaxis = "Mortality Rate, deaths/cases",
        label = nothing,
        lw = 2,
    )
    return fig1, fig2
end

function flatten_curvev(α_arr, equations, u₀, p, tspan, population)
    fig1 = plot()
    fig2 = plot()
    mortality = zeros(length(α_arr))
    for i = 1:length(α_arr)
        p[2] = α_arr[i]
        prob = ODEProblem(equations, u₀, tspan, p)
        sol = solve(prob)
        # plot!(fig1, sol, vars = 3)
        plot!(
            fig1,
            sol,
            vars=3,
            xaxis = "Time, days",
            yaxis = "Vulnerable Cases",
            label = "\\alpha = $(p[2])",
            lw = 2,
        )
        mortality[i] = sol[6, end] / sol[5, end]
    end
    plot!(
        fig2,
        α_arr,
        mortality,
        xaxis = "\\alpha",
        yaxis = "Mortality Rate, deaths/cases",
        label = nothing,
        lw = 2,
    )
    return fig1, fig2
end

raw = CSV.read("input.csv"; transpose = true, ignoreemptylines = false)
df = dropmissing(raw)

# Basic SIRD
u₀, p, tspan, population = setup(df)
@manipulate for tmax = 100:10:300, latent_period = 0:2:20, mortality = 0.01:0.01:0.15, r0 = 2:0.1:3
    γ = 1 / latent_period
    μ = γ * mortality
    λ = r0 * (γ + μ)
    p = [λ; γ; μ]
    tspan = [0.; tmax]
    prob = ODEProblem(sird!, u₀, tspan, p)
    sol = solve(prob)
    sird = plot(
        sol,
        xaxis = "Time, days",
        yaxis = "Cases",
        label = ["Susceptible" "Infected" "Recovered" "Deaths"],
        lw = 2,
    )
end

# Single event SIRD
u₀, p, tspan, population = setup(df)
@manipulate for tq = 0:5:150, q0 = 0:0.05:1
    p[3] = p[2] * 0.05
    p[end-2:end-1] = [tq; q0]
    tspan = [0.; 300]
    prob = ODEProblem(sirdq!, u₀, tspan, p)
    sol = solve(prob)
    # event_sird = plot(sol)
    event_sird = plot(
        sol,
        xaxis = "Time, days",
        yaxis = "Cases",
        label = ["Susceptible" "Infected" "Recovered" "Deaths"],
        lw = 2,
    )
    plot!(
        event_sird,
        p[7:end-2],
        linetype = :vline,
        lw = 2,
        ls = :dash,
        lc = "black",
        label = "Quarantine Event",
    )
end

# Two event SIRD
u₀, p, tspan, population = setup(raw)
@manipulate for tq1 = 0:5:300, tq2 = 0:5:300, q0 = 0:0.05:1
    p[3] = p[2] * 0.05
    p[end-3:end-1] = [tq1; tq2; q0]
    prob = ODEProblem(sirdq!, u₀, tspan, p)
    sol = solve(prob)
    # two_event_sird = plot(sol)
    two_event_sird = plot(
        sol,
        xaxis = "Time, days",
        yaxis = "Cases",
        label = ["Susceptible" "Infected" "Recovered" "Deaths"],
        lw = 2,
    )
    plot!(
        two_event_sird,
        p[7:end-2],
        linetype = :vline,
        lw = 2,
        ls = :dash,
        lc = "black",
        label = "Quarantine Event",
    )
end

# Beds single event
u₀, p, tspan, population = setup(df)
@manipulate for tq = 0:5:300, q0 = 0:0.05:1, serious = 0.1:0.01:0.5, beds=0.001:0.001:0.05, β = 10:10:100
    p[4:end-1] = [serious; beds; β; tq; q0]
    prob = ODEProblem(sirdqb!, u₀, tspan, p)
    sol = solve(prob)
    # bed_sird = plot(sol)
    bed_sird = plot(
        sol,
        xaxis = "Time, days",
        yaxis = "Cases",
        label = ["Susceptible" "Infected" "Recovered" "Deaths"],
        lw = 2,
    )
    plot!(
        bed_sird,
        p[7:end-2],
        linetype = :vline,
        lw = 2,
        ls = :dash,
        lc = "black",
        label = "Quarantine Event",
    )
    plot!(
        bed_sird,
        p[5:5],
        linetype = :hline,
        lw = 2,
        ls = :dash,
        lc = "red",
        label = "Critical Care Beds",
    )
end

# Beds two events
u₀, p, tspan, population = setup(raw)
@manipulate for tq1 = 0:5:300, tq2 = 0:5:300, q0 = 0:0.05:1, serious = 0.1:0.05:0.5, beds=0.001:0.001:0.05, β = 10:10:100
    p[4:end-1] = [serious; beds; β; tq1; tq2; q0]
    prob = ODEProblem(sirdqb!, u₀, tspan, p)
    sol = solve(prob)
    # bed_sird = plot(sol)
    bed_sird = plot(
        sol,
        xaxis = "Time, days",
        yaxis = "Cases",
        label = ["Susceptible" "Infected" "Recovered" "Deaths"],
        lw = 2,
    )
    plot!(
        bed_sird,
        p[7:end-2],
        linetype = :vline,
        lw = 2,
        ls = :dash,
        lc = "black",
        label = "Quarantine Event",
    )
    plot!(
        bed_sird,
        p[5:5],
        linetype = :hline,
        lw = 2,
        ls = :dash,
        lc = "red",
        label = "Critical Care Beds",
    )
end

# Flatten curve
u₀, p, tspan, population = setup(df)
p[7:end-2] .= 0.0
q_arr = Array(range(0, step = 0.1, stop = 0.6))
flat_curve, mort_plot = flatten_curve(q_arr, sirdqb!, u₀, p, tspan, population)
plot!(
    flat_curve,
    p[5:5],
    linetype = :hline,
    lw = 2,
    ls = :dash,
    lc = "red",
    label = "Critical Care Beds",
)
display(flat_curve)
display(mort_plot)

# sird vulnerable
u₀, p, tspan, population, S₀, I₀ = setupv(raw)
@manipulate for mortalityv = 0.05:0.01:0.20, mortalitynv = 0.001:0.001:0.01, fv = 0:0.01:1, α = 0:0.1:1
    μv = p[3] * mortalityv
    μnv = p[3] * mortalitynv
    μ = μv * fv + μnv * (1 - fv)
    λ = 2.6 * (p[3] + μ)
    p[1] = λ
    p[2] = α
    p[4:5] = [μv; μnv]

    Sv₀ = fv * S₀
    Snv₀ = (1 - fv) * S₀
    Iv₀ = fv * I₀
    Inv₀ = (1 - fv) * I₀
    u₀[1:4] = [Sv₀; Snv₀; Iv₀; Inv₀]

    prob = ODEProblem(sirdv!, u₀, tspan, p)
    sol = solve(prob)
    sirdv = plot(sol)
    sirdv = plot(
        sol,
        xaxis = "Time, days",
        yaxis = "Cases",
        label = ["Susceptible Vulnerable" "Susceptible Not Vulnerable" "Infected Vulnerable" "Infected Not Vulnerable" "Recovered" "Deaths"],
        lw = 2,
    )
end

# Flatten sird vulnerable
u₀, p, tspan, population = setupv(raw)
α_arr = Array(range(0, step = 0.2, stop = 0.9))
flat_curve, mort_plot = flatten_curvev(α_arr, sirdv!, u₀, p, tspan, population)
plot!(
    flat_curve,
    p[5:5],
    linetype = :hline,
    lw = 2,
    ls = :dash,
    lc = "red",
    label = "Critical Care Beds",
)
display(flat_curve)
display(mort_plot)
